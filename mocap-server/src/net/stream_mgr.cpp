#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sched.h>
#include <signal.h>
#include <spsc_queue.hpp>
#include <sys/socket.h>
#include <unistd.h>

#include "core/queue.hpp"
#include "core/logging.h"
#include "net/network.hpp"
#include "net/stream_mgr.hpp"
#include "video/viddec.hpp"

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

  thread_ctx* ctx = static_cast<thread_ctx*>(ptr);
  uint32_t dev_ptrs_idx = 0;

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
      snprintf(logstr, sizeof(logstr),
        "Error pinning thread for cam %hhu to core %d: %s",
        ctx->conf->id, ctx->core, strerror(errno));
      log_write(ERROR, logstr);
    }
  }

  {
    queue timestamp_queue;
    ret = init_queue(&timestamp_queue, sizeof(uint64_t), TS_Q_INIT_SIZE);
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

    // UDP reassembly buffer
    reassembly_buf reasm;
    reassembly_init(&reasm);

    uint8_t udp_recv_buf[UDP_MTU];

    snprintf(logstr, sizeof(logstr), "Stream manager for cam %s ready, receiving UDP frames",
      ctx->conf->name);
    log_write(INFO, logstr);

    while (running && *ctx->main_running) {
      // receive UDP datagram
      ssize_t recv_size = recv(ctx->udp_fd, udp_recv_buf, sizeof(udp_recv_buf), 0);
      if (recv_size < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK)
          continue; // timeout, just retry
        if (errno == EINTR)
          goto shutdown_path;
        snprintf(logstr, sizeof(logstr), "UDP recv error: %s", strerror(errno));
        log_write(ERROR, logstr);
        continue;
      }

      // try to reassemble
      uint64_t timestamp = 0;
      int frame_size = reassembly_add_fragment(
        &reasm,
        udp_recv_buf,
        recv_size,
        &timestamp
      );

      if (frame_size <= 0)
        continue; // incomplete frame, keep receiving

      log_write(BENCHMARK, "Received complete UDP frame");

      // enqueue timestamp
      ret = enqueue(&timestamp_queue, static_cast<void*>(&timestamp));
      if (ret) {
        cleanup_decoder(&viddec);
        cleanup_queue(&timestamp_queue);
        goto err_cleanup;
      }

      // wait for available device ptrs before decoding
      struct timespec ts = {.tv_sec = 0, .tv_nsec = NO_DEV_PTRS_WAIT};
      while (ctx->dev_ptrs_used->load(std::memory_order_relaxed) >= ctx->dev_ptrs_total) {
        if (!running || !*ctx->main_running)
          goto shutdown_path;
        nanosleep(&ts, nullptr);
      }
      ctx->dev_ptrs_used->fetch_add(1, std::memory_order_relaxed);

      log_write(BENCHMARK, "Started decoding packet");

      ret = decode_packet(&viddec, reasm.data, frame_size);
      if (ret) {
        cleanup_decoder(&viddec);
        cleanup_queue(&timestamp_queue);
        goto err_cleanup;
      }

      log_write(BENCHMARK, "Finished decoding packet");

      // try to receive decoded frame
      ts_dev_ptr* dev_ptr = &ctx->dev_ptrs[dev_ptrs_idx];
      dev_ptrs_idx = (dev_ptrs_idx + 1) % ctx->dev_ptrs_total;

      ret = recv_frame(&viddec, &dev_ptr->dev_ptr);
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
  return nullptr;
}
