#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sched.h>
#include <signal.h>
#include <spsc_queue.hpp>
#include <unistd.h>

#include "queue.hpp"
#include "logging.h"
#include "network.hpp"
#include "stream_mgr.hpp"
#include "viddec.hpp"

#define TS_Q_INIT_SIZE 8
#define NO_DEV_PTRS_WAIT 10000000 // 10 ms

static volatile sig_atomic_t running = 1;

static void shutdown_handler(int signum) {
  (void)signum;
  running = 0;
}

void* stream_mgr_fn(void* ptr) {
  int ret = 0;
  char logstr[128];

  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = shutdown_handler;
  sa.sa_flags = 0;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGUSR2, &sa, nullptr);

  int sockfd = -1;
  int clientfd = -1;

  thread_ctx* ctx = static_cast<thread_ctx*>(ptr);
  uint32_t dev_ptrs_idx = 0; // wraps around at ctx->dev_ptrs_total

  uint8_t* enc_frame_buf = static_cast<uint8_t*>(malloc(ENCODED_FRAME_BUF_SIZE));
  if (!enc_frame_buf) {
    log_write(ERROR, "Failed to allocate encoded frame buffer in a thread");
    goto err_cleanup;
  }

  {
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
        "Error pinning thread for cam %hhu to core %d, err: %s",
        ctx->conf->id,
        ctx->core,
        strerror(errno)
      );
      log_write(ERROR, logstr);
      ret = -errno;
      goto err_cleanup;
    }
  }

  {
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
      ctx->strm_conf->frame_width,
      ctx->strm_conf->frame_height,
      ctx->dev_ptrs_total,
      ctx->hw_device_ctx
    );
    if (ret) {
      cleanup_queue(&timestamp_queue);
      goto err_cleanup;
    }

    sockfd = setup_stream(ctx->conf);
    if (sockfd < 0) {
      ret = -EIO;
      cleanup_decoder(&viddec);
      cleanup_queue(&timestamp_queue);
      goto err_cleanup;
    }

    clientfd = accept_conn(sockfd);
    if (clientfd < 0) {
      ret = -EIO;
      cleanup_decoder(&viddec);
      cleanup_queue(&timestamp_queue);
      goto err_cleanup;
    }

    bool incoming_stream = true;
    while (running && *ctx->main_running) {
      if (incoming_stream) {
        uint64_t timestamp = 0;
        ssize_t pkt_size = recv_from_stream(
          clientfd,
          reinterpret_cast<char*>(&timestamp),
          sizeof(timestamp)
        );

        if (pkt_size != sizeof(timestamp)) {
          if (errno == EINTR)
            goto shutdown_path;

          snprintf(
            logstr,
            sizeof(logstr),
            "Received unexpected timestamp size %zd from cam %s",
            pkt_size,
            ctx->conf->name
          );
          log_write(ERROR, logstr);
          cleanup_decoder(&viddec);
          cleanup_queue(&timestamp_queue);
          goto err_cleanup;
        }

        if (memcmp(&timestamp, "EOSTREAM", 8) == 0) {
          incoming_stream = false;
          ret = flush_decoder(&viddec);
          if (ret) {
            cleanup_decoder(&viddec);
            cleanup_queue(&timestamp_queue);
            goto shutdown_path;
          }
          continue;
        }

        log_write(BENCHMARK, "Received packet header");

        ret = enqueue(&timestamp_queue, static_cast<void*>(&timestamp));
        if (ret) {
          cleanup_decoder(&viddec);
          cleanup_queue(&timestamp_queue);
          goto err_cleanup;
        }

        uint32_t frame_size = 0;
        pkt_size = recv_from_stream(
          clientfd,
          reinterpret_cast<char*>(&frame_size),
          sizeof(frame_size)
        );

        if (pkt_size != sizeof(frame_size)) {
          if (errno == EINTR)
            goto shutdown_path;

          snprintf(
            logstr,
            sizeof(logstr),
            "Received unexpected frame size %zd from cam %s",
            pkt_size,
            ctx->conf->name
          );
          log_write(ERROR, logstr);
          cleanup_decoder(&viddec);
          cleanup_queue(&timestamp_queue);
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
          log_write(ERROR, logstr);
          cleanup_decoder(&viddec);
          cleanup_queue(&timestamp_queue);
          goto err_cleanup;
        }

        pkt_size = recv_from_stream(
          clientfd,
          reinterpret_cast<char*>(enc_frame_buf),
          frame_size
        );

        if (pkt_size != frame_size) {
          if (errno == EINTR)
            goto shutdown_path;

          snprintf(
            logstr,
            sizeof(logstr),
            "Received unexpected frame size with %zd bytes from cam %s",
            pkt_size,
            ctx->conf->name
          );
          log_write(ERROR, logstr);
          cleanup_decoder(&viddec);
          cleanup_queue(&timestamp_queue);
          goto err_cleanup;
        }

        log_write(BENCHMARK, "Received full packet");
        log_write(BENCHMARK, "Started decoding packet");

        // wait for available device ptrs before decoding
        struct timespec ts = {
          .tv_sec = 0,
          .tv_nsec = NO_DEV_PTRS_WAIT
        };
        while (ctx->dev_ptrs_used->load(std::memory_order_relaxed) >= ctx->dev_ptrs_total) {
          if (!running || !*ctx->main_running)
            goto shutdown_path;
          nanosleep(&ts, nullptr);
        }
        ctx->dev_ptrs_used->fetch_add(1, std::memory_order_relaxed);

        ret = decode_packet(
          &viddec,
          enc_frame_buf,
          frame_size
        );
        if (ret) {
          cleanup_decoder(&viddec);
          cleanup_queue(&timestamp_queue);
          goto err_cleanup;
        }

        log_write(BENCHMARK, "Finished decoding packet");
      }

      ts_dev_ptr* dev_ptr = &ctx->dev_ptrs[dev_ptrs_idx];
      dev_ptrs_idx += 1;
      if (dev_ptrs_idx >= ctx->dev_ptrs_total)
        dev_ptrs_idx = 0;

      ret = recv_frame(
        &viddec,
        &dev_ptr->dev_ptr
      );

      if (ret == EAGAIN) {
        continue;
      } else if (ret) {
        cleanup_decoder(&viddec);
        cleanup_queue(&timestamp_queue);
        goto err_cleanup;
      } else {
        dequeue(&timestamp_queue, static_cast<void*>(&dev_ptr->timestamp));
        spsc_enqueue(ctx->dev_ptr_queue, static_cast<void*>(dev_ptr));
      }
    }

    shutdown_path:
    cleanup_decoder(&viddec);
    cleanup_queue(&timestamp_queue);
    goto done;

  }

err_cleanup:
  *ctx->main_running = 0;

done:
  if (enc_frame_buf)
    free(enc_frame_buf);
  if (sockfd >= 0)
    close(sockfd);
  if (clientfd >= 0)
    close(clientfd);

  return nullptr;
}
