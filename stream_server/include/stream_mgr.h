#ifndef STREAM_MGR_H
#define STREAM_MGR_H

#include <semaphore.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

#include "parse_conf.h"

#define ENCODED_FRAME_BUF_SIZE 6400 // 6.4kb
#define DECODED_FRAME_WIDTH 1080
#define DECODED_FRAME_HEIGHT 720

typedef struct thread_ctx {
  cam_conf* conf;
  uint32_t core;
  uint32_t frames_total;
  atomic_uint_least32_t* frames_filled;
  atomic_bool* new_frame;
  uint64_t* timestamp;
  uint8_t* frame_buf;
  sem_t* loop_ctl_sem;
} thread_ctx;

void* stream_mgr(void* ptr);

#endif // STREAM_MGR_H
