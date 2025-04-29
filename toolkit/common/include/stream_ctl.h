#ifndef STREAM_CTL_H
#define STREAM_CTL_H

#include <cstdint>
#include <sys/types.h>

#include "spsc_queue.hpp"

// the following constants need to match the stream server identically:
constexpr const char* SERVER_EXE = "/usr/local/bin/mocap-toolkit-server";
constexpr const char* SHM_NAME = "/mocap-toolkit_shm";
#define SHM_ADDR ((void*)0x7f0000000000)
#define DEV_PTRS_PER_THREAD 16

struct stream_ctx {
  pid_t server_pid;
  int32_t shm_fd;
  uint64_t shm_size;
  void* mmap_buf;
  consumer_q* ipc_handles_cq;
  std::atomic<uint32_t>* counters;
};

int32_t start_streams(
  stream_ctx& ctx,
  uint32_t cam_count,
  char* target_id
);
void cleanup_streams(stream_ctx& ctx);

#endif // STREAM_CTL_H
