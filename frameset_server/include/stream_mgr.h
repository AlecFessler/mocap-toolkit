#ifndef STREAM_MGR_H
#define STREAM_MGR_H

#include <signal.h>
#include <spsc_queue.h>
#include <stdint.h>

#include "parse_conf.h"

#define ENCODED_FRAME_BUF_SIZE 96000
#define DECODED_FRAME_WIDTH 1280
#define DECODED_FRAME_HEIGHT 720

struct thread_ctx {
  cam_conf* conf;
  struct producer_q* filled_bufs;
  struct consumer_q* empty_bufs;
  uint32_t core;
  volatile sig_atomic_t* main_running;
};

struct ts_frame_buf {
  uint64_t timestamp;
  uint8_t* frame_buf;
};

void* stream_mgr_fn(void* ptr);

#endif // STREAM_MGR_H
