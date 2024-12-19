#define _GNU_SOURCE
#include <errno.h>
#include <sched.h>
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

void* stream_mgr(void* ptr) {
  int ret = 0;
  char logstr[128];

  struct thread_ctx* ctx = (struct thread_ctx*)ptr;
  uint8_t* enc_frame_buf = malloc(ENCODED_FRAME_BUF_SIZE);
  if (!enc_frame_buf) {
    log(ERROR, "Failed to allocate encoded frame buffer in a thread");
    goto cleanup;
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(ctx->core, &cpuset);

  pid_t tid = gettid();
  ret = sched_setaffinity(
    tid,
    sizeof(cpu_set_t),
    &cpuset
  );
  if (ret == -1) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error pinning thread %d to core %d, err: %s",
      tid,
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

  int sockfd = setup_stream(ctx->conf);
  if (sockfd < 0) {
    ret = -EIO;
    goto cleanup;
  }

  int clientfd = accept_conn(sockfd);
  if (clientfd < 0) {
    ret = -EIO;
    goto cleanup;
  }

  struct ts_frame_buf* current_buf = (struct ts_frame_buf*)spsc_dequeue(ctx->empty_bufs);

  bool incoming_stream = true;
  while (true) {
    if (incoming_stream) {
      uint64_t timestamp = 0;
      ssize_t pkt_size = recv_from_stream(
        clientfd,
        (char*)&timestamp,
        sizeof(timestamp)
      );

      if (pkt_size != sizeof(timestamp)) {
        snprintf(
          logstr,
          sizeof(logstr),
          "Received unexpected timestamp size with %zd bytes from cam %s",
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
        snprintf(
          logstr,
          sizeof(logstr),
          "Received unexpected frame size buffer with %ld bytes from cam %s",
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
  if (enc_frame_buf) {
    free(enc_frame_buf);
  }
  cleanup_decoder(&viddec);
  cleanup_queue(&timestamp_queue);
  if (sockfd >= 0) {
    close(sockfd);
  }
  if (clientfd >= 0) {
    close(clientfd);
  }

  return NULL;
}
