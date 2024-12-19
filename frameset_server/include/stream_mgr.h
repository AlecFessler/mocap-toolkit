#ifndef STREAM_MGR_H
#define STREAM_MGR_H

#include <stdint.h>

#include "parse_conf.h"
#include "spsc_queue.h"

#define ENCODED_FRAME_BUF_SIZE 6400 // 6.4kb
#define DECODED_FRAME_WIDTH 1080
#define DECODED_FRAME_HEIGHT 720

struct thread_ctx {
  cam_conf* conf;
  struct producer_q* filled_bufs;
  struct consumer_q* empty_bufs;
  uint32_t core;
};

struct ts_frame_buf {
  uint64_t timestamp;
  uint8_t* frame_buf;
};

void* stream_mgr_fn(void* ptr);

#endif // STREAM_MGR_H
