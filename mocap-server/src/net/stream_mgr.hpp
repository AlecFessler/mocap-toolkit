#ifndef STREAM_MGR_HPP
#define STREAM_MGR_HPP

#include <atomic>
#include <cstdint>
#include <signal.h>
#include <spsc_queue.hpp>

#include "core/parse_conf.h"

extern "C" {
#include <libavutil/hwcontext.h>
}

#define ENCODED_FRAME_BUF_SIZE 262144 // 256KB

struct ts_dev_ptr {
  uint64_t timestamp;
  void* dev_ptr;
};

struct thread_ctx {
  cam_conf* conf;
  struct stream_conf* strm_conf;
  producer_q* dev_ptr_queue;
  std::atomic<uint32_t>* dev_ptrs_used;
  uint32_t dev_ptrs_total;
  ts_dev_ptr* dev_ptrs;
  uint32_t core;
  volatile sig_atomic_t* main_running;
  AVBufferRef* hw_device_ctx;
  int ctrl_fd;   // TCP control connection (already accepted)
  int udp_fd;    // UDP receive socket (already bound)
};

void* stream_mgr_fn(void* ptr);

#endif // STREAM_MGR_HPP
