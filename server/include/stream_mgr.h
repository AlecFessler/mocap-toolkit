#ifndef STREAM_MGR_H
#define STREAM_MGR_H

#include <signal.h>
#include <spsc_queue.h>
#include <stdatomic.h>
#include <stdint.h>

#include "parse_conf.h"

#define ENCODED_FRAME_BUF_SIZE 262144 // 256KB

struct thread_ctx {
  struct cam_conf* conf;
  struct stream_conf* stream_conf;
  struct producer_q* dev_ptr_queue;
  _Atomic uint32_t* dev_ptrs_used;
  uint32_t dev_ptrs_total;
  struct ts_dev_ptr* dev_ptrs;
  uint32_t core;
  volatile sig_atomic_t* main_running;
};

struct ts_dev_ptr {
  uint64_t timestamp;
  void* dev_ptr;
};

void* stream_mgr_fn(void* ptr);

#endif // STREAM_MGR_H
