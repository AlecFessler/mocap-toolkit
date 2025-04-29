#define _GNU_SOURCE
#include <errno.h>
#include <sched.h>
#include <signal.h>
#include <spsc_queue.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "queue.h"
#include "logging.h"
#include "network.h"
#include "stream_mgr.h"
#include "viddec.h"

#define TS_Q_INIT_SIZE 8
#define NO_DEV_PTRS_WAIT 10000 // 0.01 ms

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
  uint32_t dev_ptrs_idx = 0; // wraps around at ctx->dev_ptrs_total

  uint8_t* enc_frame_buf = malloc(ENCODED_FRAME_BUF_SIZE);
  if (!enc_frame_buf) {
    log(ERROR, "Failed to allocate encoded frame buffer in a thread");
    goto err_cleanup;
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
    goto err_cleanup;
  }

  queue timestamp_queue;
  ret = init_queue(
    &timestamp_queue,
    sizeof(uint64_t),
    TS_Q_INIT_SIZE
  );
  if (ret)
    goto err_cleanup;

  decoder viddec;
  ret = init_decoder(
    &viddec,
    ctx->stream_conf->frame_width,
    ctx->stream_conf->frame_height,
    ctx->dev_ptrs_total
  );
  if (ret)
    goto err_cleanup;

  sockfd = setup_stream(ctx->conf);
  if (sockfd < 0) {
    ret = -EIO;
    goto err_cleanup;
  }

  clientfd = accept_conn(sockfd);
  if (clientfd < 0) {
    ret = -EIO;
    goto err_cleanup;
  }

  bool incoming_stream = true;
  while (running && ctx->main_running) {
    if (incoming_stream) {
      uint64_t timestamp = 0;
      ssize_t pkt_size = recv_from_stream(
        clientfd,
        (char*)&timestamp,
        sizeof(timestamp)
      );

      if (pkt_size != sizeof(timestamp)) {
        if (errno == -EINTR)
          goto shutdown_cleanup;

        snprintf(
          logstr,
          sizeof(logstr),
          "Received unexpected timestamp size %zd from cam %s",
          pkt_size,
          ctx->conf->name
        );
        log(ERROR, logstr);
        goto err_cleanup;
      }

      if (memcmp(&timestamp, "EOSTREAM", 8) == 0) {
        incoming_stream = false;
        ret = flush_decoder(&viddec);
        if (ret)
          goto shutdown_cleanup;

        continue;
      }

      log(BENCHMARK, "Received packet header");

      ret = enqueue(&timestamp_queue, (void*)&timestamp);
      if (ret)
        goto err_cleanup;

      uint32_t frame_size = 0;
      pkt_size = recv_from_stream(
        clientfd,
        (char*)&frame_size,
        sizeof(frame_size)
      );

      if (pkt_size != sizeof(frame_size)) {
        if (errno == -EINTR)
          goto shutdown_cleanup;

        snprintf(
          logstr,
          sizeof(logstr),
          "Received unexpected frame size %ld from cam %s",
          pkt_size,
          ctx->conf->name
        );
        log(ERROR, logstr);
        goto err_cleanup;
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
        goto err_cleanup;
      }

      pkt_size = recv_from_stream(
        clientfd,
        (char*)enc_frame_buf,
        frame_size
      );

      if (pkt_size != frame_size) {
        if (errno == -EINTR)
          goto shutdown_cleanup;

        snprintf(
          logstr,
          sizeof(logstr),
          "Received unexpected frame size with %zd bytes from cam %s",
          pkt_size,
          ctx->conf->name
        );
        log(ERROR, logstr);
        goto err_cleanup;
      }

      log(BENCHMARK, "Received full packet");
      log(BENCHMARK, "Started decoding packet");

      // check for available device ptrs before decoding
      struct timespec ts = {
        .tv_sec = 0,
        .tv_nsec = NO_DEV_PTRS_WAIT
      };
      while (atomic_load_explicit(ctx->dev_ptrs_used, memory_order_relaxed) >= ctx->dev_ptrs_total) {
        nanosleep(&ts, NULL);
      }
      atomic_fetch_add(ctx->dev_ptrs_used, 1);

      ret = decode_packet(
        &viddec,
        enc_frame_buf,
        frame_size
      );
      if (ret)
        goto err_cleanup;

      log(BENCHMARK, "Finished decoding packet");
    }

    struct ts_dev_ptr* dev_ptr = &ctx->dev_ptrs[dev_ptrs_idx];
    dev_ptrs_idx += 1;
    if (dev_ptrs_idx >= ctx->dev_ptrs_total) {
      dev_ptrs_idx = 0;
    }
    ret = recv_frame(
      &viddec,
      &dev_ptr->dev_ptr
    );

    if (ret == EAGAIN) {
      continue;
    } else if (ret) {
      goto err_cleanup;
    } else {
      dequeue(&timestamp_queue, (void*)&dev_ptr->timestamp);
      spsc_enqueue(ctx->dev_ptr_queue, (void*)dev_ptr);
    }
  }

err_cleanup:
  ctx->main_running = 0;

shutdown_cleanup:
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
