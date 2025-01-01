#ifndef VID_PLAYER_HPP
#define VID_PLAYER_HPP

#include <cstdint>
#include <spsc_queue.hpp>

#include "parse_conf.h"

struct display_thread_ctx {
  struct consumer_q* filled_frameset_q;
  struct producer_q* empty_frameset_q;
  struct stream_conf* stream_conf;
  uint32_t num_frames;
  uint32_t core;
};

void* display_thread_fn(void* ptr);

#endif //  VID_PLAYER_HPP
