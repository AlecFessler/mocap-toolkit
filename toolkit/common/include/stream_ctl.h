#ifndef STREAM_CTL_H
#define STREAM_CTL_H

#include <cstdint>
#include <sys/types.h>

#include "spsc_queue.hpp"

// the following constants need to match the stream server identically:
constexpr const char* SERVER_EXE = "/usr/local/bin/mocap-toolkit-server";
constexpr const char* SHM_NAME = "/mocap-toolkit_shm";
#define SHM_ADDR ((void*)0x7f0000000000)
constexpr uint32_t FRAME_BUFS_PER_THREAD = 512;
constexpr uint32_t FRAMESET_SLOTS_PER_THREAD = 8;
constexpr uint32_t FRAME_WIDTH = 1280;
constexpr uint32_t FRAME_HEIGHT = 720;
constexpr uint64_t FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 3 / 2; // nv12 fmt

struct stream_ctx {
  pid_t server_pid;
  int32_t shm_fd;
  uint64_t shm_size;
  void* mmap_buf;
  consumer_q* filled_frameset_q;
  producer_q* empty_frameset_q;
};

struct ts_frame_buf {
  uint64_t timestamp;
  uint8_t* frame_buf;
};

int32_t start_streams(stream_ctx& ctx, uint32_t cam_count);
void cleanup_streams(stream_ctx& ctx);

#endif // STREAM_CTL_H
