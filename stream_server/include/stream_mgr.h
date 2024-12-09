#ifndef STREAM_MGR_H
#define STREAM_MGR_H

#include <semaphore.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

#include "parse_conf.h"

#define ENCODED_FRAME_BUF_SIZE 16000 // 16kb
#define DECODED_FRAME_BUF_SIZE 2764800 // 1280 * 720 * 3 (720p RGB24)

typedef struct thread_ctx {
  atomic_uint_least8_t* frames_filled;
  atomic_bool* new_frame;
  sem_t* loop_ctl_sem;
  cam_conf* conf;
  uint8_t* frame_buf;
  uint64_t timestamp;
  uint8_t thread_idx;
  uint8_t core;
  uint8_t frames_total;
} thread_ctx;

int pin_to_core(int core);
void* stream_mgr(void* ptr);

#endif // STREAM_MGR_H
