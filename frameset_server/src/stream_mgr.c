#define _GNU_SOURCE
#include <errno.h>
#include <sched.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "queue.h"
#include "spsc_queue.h"
#include "logging.h"
#include "network.h"
#include "stream_mgr.h"
#include "viddec.h"

static volatile sig_atomic_t running = 1;

static void shutdown_handler(int signum);

void* stream_mgr_fn(void* ptr) {
  int ret = 0;
  char logstr[128];

  struct sigaction sa = {
    .sa_handler = shutdown_handler,
    .sa_flags = 0
  };
  sigemptyset(&sa.sa_mask);
  sigaction(SIGUSR2, &sa, NULL);

  int sockfd = -1;
  int clientfd = -1;

  struct thread_ctx* ctx = (struct thread_ctx*)ptr;
  uint8_t* enc_frame_buf = malloc(ENCODED_FRAME_BUF_SIZE);
  if (!enc_frame_buf) {
    log(ERROR, "Failed to allocate encoded frame buffer in a thread");
    goto cleanup;
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(ctx->core, &cpuset);
  ret = sched_setaffinity(
    gettid(),
    sizeof(cpu_set_t),
    &cpuset
  );
  if (ret == -1) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error pinning thread %d to core %d, err: %s",
      gettid(),
      ctx->core,
      strerror(errno)
    );
    log(ERROR, logstr);
    ret = -errno;
    goto cleanup;
  }

  queue timestamp_queue;
  ret = init_queue(
    &timestamp_queue,
    sizeof(uint64_t),
    8 // initial capacity
  );
  if (ret) {
    goto cleanup;
  }

  decoder viddec;
  ret = init_decoder(
    &viddec,
    DECODED_FRAME_WIDTH,
    DECODED_FRAME_HEIGHT
  );
  if (ret) {
    goto cleanup;
  }

  sockfd = setup_stream(ctx->conf);
  if (sockfd < 0) {
    ret = -EIO;
    goto cleanup;
  }

  clientfd = accept_conn(sockfd);
  if (clientfd < 0) {
    ret = -EIO;
    goto cleanup;
  }

  struct ts_frame_buf* current_buf = (struct ts_frame_buf*)spsc_dequeue(ctx->empty_bufs);

  bool incoming_stream = true;
  while (running) {
    if (incoming_stream) {
      uint64_t timestamp = 0;
      ssize_t pkt_size = recv_from_stream(
        clientfd,
        (char*)&timestamp,
        sizeof(timestamp)
      );

      if (pkt_size != sizeof(timestamp)) {
        if (errno == EINTR)
          break; // shutdown signal

        snprintf(
          logstr,
          sizeof(logstr),
          "Received unexpected timestamp size %zd from cam %s",
          pkt_size,
          ctx->conf->name
        );
        log(ERROR, logstr);
        break;
      }

      if (memcmp(&timestamp, "EOSTREAM", 8) == 0) {
        log(INFO, "Received end of stream signal");
        incoming_stream = false;
        ret = flush_decoder(&viddec);
        if (ret) {
          break;
        }
        continue;
      }

      ret = enqueue(&timestamp_queue, (void*)&timestamp);
      if (ret) {
        break;
      }

      uint32_t frame_size = 0;
      pkt_size = recv_from_stream(
        clientfd,
        (char*)&frame_size,
        sizeof(frame_size)
      );

      if (pkt_size != sizeof(frame_size)) {
        if (errno == EINTR)
          break; // shutdown signal

        snprintf(
          logstr,
          sizeof(logstr),
          "Received unexpected frame size %ld from cam %s",
          pkt_size,
          ctx->conf->name
        );
        log(ERROR, logstr);
        break;
      }

      if (frame_size > ENCODED_FRAME_BUF_SIZE) {
        snprintf(
          logstr,
          sizeof(logstr),
          "Received frame size that is larger than the allocated buffer of %d bytes: %d",
          ENCODED_FRAME_BUF_SIZE,
          frame_size
        );
        log(ERROR, logstr);
        break;
      }

      pkt_size = recv_from_stream(
        clientfd,
        (char*)enc_frame_buf,
        frame_size
      );

      if (pkt_size != frame_size) {
        if (errno == EINTR)
          break; // shutdown signal

        snprintf(
          logstr,
          sizeof(logstr),
          "Received unexpected frame size with %zd bytes from cam %s",
          pkt_size,
          ctx->conf->name
        );
        log(ERROR, logstr);
        break;
      }

      ret = decode_packet(
        &viddec,
        enc_frame_buf,
        frame_size
      );
      if (ret) {
        break;
      }
    }

    ret = recv_frame(
      &viddec,
      current_buf->frame_buf
    );

    if (ret == EAGAIN) {
      continue;
    } else if (ret == ENODATA) {
      log(INFO, "Received EOF from decoder");
      break;
    } else if (ret) {
      break;
    } else {
      dequeue(&timestamp_queue, (void*)&current_buf->timestamp);
      spsc_enqueue(ctx->filled_bufs, (void*)current_buf);

      current_buf = (struct ts_frame_buf*)spsc_dequeue(ctx->empty_bufs);
      if (!current_buf) {
        log(ERROR, "Frame buffer queue was empty");
        ret = -ENOBUFS;
        goto cleanup;
      }
    }
  }

cleanup:
  if (enc_frame_buf)
    free(enc_frame_buf);
  cleanup_decoder(&viddec);
  cleanup_queue(&timestamp_queue);
  if (sockfd >= 0)
    close(sockfd);
  if (clientfd >= 0)
    close(clientfd);

  return NULL;
}

static void shutdown_handler(int signum) {
  (void)signum;
  running = 0;
}
