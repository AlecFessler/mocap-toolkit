#include <csignal>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cerrno>
#include <fcntl.h>
#include <memory>
#include <spsc_queue.hpp>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "stream_ctl.h"
#include "logging.h"

static inline uint64_t alignup(uint64_t offset, uint64_t alignment) {
  return (offset + (alignment - 1)) & ~(alignment - 1);
}

int32_t start_streams(
  struct stream_ctx& ctx,
  uint32_t frame_width,
  uint32_t frame_height,
  uint32_t cam_count,
  char* target_id = nullptr
) {
  char logstr[128];

  ctx.server_pid = 0;
  ctx.shm_fd = -1;
  ctx.shm_size = 0;
  ctx.mmap_buf = nullptr;
  ctx.filled_frameset_q = nullptr;
  ctx.empty_frameset_q = nullptr;

  ctx.server_pid = fork();
  if (ctx.server_pid == -1) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Failed to fork process: %s",
      strerror(errno)
    );
    log_write(ERROR, logstr);
    cleanup_streams(ctx);
    return -errno;
  }

  if (ctx.server_pid == 0) {
    execl(SERVER_EXE, SERVER_EXE, target_id, nullptr);
    _exit(errno);
  }

  /**
   * The frameset producer server will create, resize, initialize, and cleanup
   * the shared memory. We just need to map out the layout so we can find our
   * filled frameset consumer queue, and empty frameset producer queues, and
   * get the size so we can unmap it at cleanup time
   */

  uint32_t max_attempts = 10;
  useconds_t retry_cd = 1000; // 1ms
  for (uint32_t attempt = 0; attempt < max_attempts; attempt++) {
    ctx.shm_fd = shm_open(
      SHM_NAME,
      O_RDWR,
      0666
    );
    if (ctx.shm_fd != -1)
      break;

    if (errno == ENOENT) {
      usleep(retry_cd);
      retry_cd *= 2;
    } else {
      break;
    }
  }

  if (ctx.shm_fd == -1) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error creating shared memory: %s",
      strerror(errno)
    );
    log_write(ERROR, logstr);
    cleanup_streams(ctx);
    return -errno;
  }

  struct stat sb;
  retry_cd = 1000; // 1ms
  for (uint32_t attempt = 0; attempt < max_attempts; attempt++) {
    int32_t ret = fstat(ctx.shm_fd, &sb);
    if (ret == -1) {
      snprintf(
        logstr,
        sizeof(logstr),
        "Error checking shared memory size: %s",
        strerror(errno)
      );
      log_write(ERROR, logstr);
      cleanup_streams(ctx);
      return -errno;
    }

    if (sb.st_size > 0)
      break;

    usleep(retry_cd);
    retry_cd *= 2;
  }

  if (sb.st_size == 0) {
    log_write(ERROR, "Timeout waiting for shared memory initialization");
    cleanup_streams(ctx);
    return -ETIMEDOUT;
  }

  // frame buffers
  uint64_t frame_size = frame_width * frame_height * 3 / 2;
  uint64_t frame_count = cam_count * FRAME_BUFS_PER_THREAD;
  ctx.shm_size = frame_size * frame_count;

  // timestamped structs with frame buffer ptrs
  ctx.shm_size = alignup(ctx.shm_size, alignof(struct ts_frame_buf));
  ctx.shm_size += sizeof(ts_frame_buf) * frame_count;

  // filled frameset producer queue
  ctx.shm_size = alignup(ctx.shm_size, alignof(struct producer_q));
  ctx.shm_size += sizeof(struct producer_q);

  // filled frameset consumer queue, we want a ref to this one
  ctx.shm_size = alignup(ctx.shm_size, alignof(struct consumer_q));
  uint64_t filled_frameset_consumer_q_offset = ctx.shm_size;
  ctx.shm_size += sizeof(struct consumer_q);

  // empty frameset producer queue, we want a ref to this one too
  ctx.shm_size = alignup(ctx.shm_size, alignof(struct producer_q));
  uint64_t empty_frameset_producer_q_offset = ctx.shm_size;
  ctx.shm_size += sizeof(struct producer_q);

  // empty frameset consumer queue
  ctx.shm_size = alignup(ctx.shm_size, alignof(struct consumer_q));
  ctx.shm_size += sizeof(struct consumer_q);

  // filled frameset queue buffer
  ctx.shm_size = alignup(ctx.shm_size, alignof(void**));
  ctx.shm_size += (sizeof(void*) * FRAME_BUFS_PER_THREAD);

  // empty frameset queue buffer
  ctx.shm_size = alignup(ctx.shm_size, alignof(void**));
  ctx.shm_size += (sizeof(void*) * FRAME_BUFS_PER_THREAD);

  // frameset slots
  ctx.shm_size = alignup(ctx.shm_size, alignof(struct ts_frame_buf*));
  ctx.shm_size += sizeof(struct ts_frame_buf*) * (FRAMESET_SLOTS_PER_THREAD * cam_count);

  if (static_cast<uint64_t>(sb.st_size) != ctx.shm_size) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Shared memory size mismatch: expected %lu, got %lu",
      ctx.shm_size,
      sb.st_size
    );
    log_write(ERROR, logstr);
    cleanup_streams(ctx);
    return -EINVAL;
  }

  ctx.mmap_buf = mmap(
    SHM_ADDR,
    ctx.shm_size,
    PROT_READ | PROT_WRITE,
    MAP_SHARED,
    ctx.shm_fd,
    0
  );
  if (ctx.mmap_buf == MAP_FAILED) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error mapping shared memory: %s",
      strerror(errno)
    );
    log_write(ERROR, logstr);
    cleanup_streams(ctx);
    return -errno;
  }

  uint8_t* base_ptr = reinterpret_cast<uint8_t*>(ctx.mmap_buf);
  ctx.filled_frameset_q = reinterpret_cast<consumer_q*>((base_ptr + filled_frameset_consumer_q_offset));
  ctx.empty_frameset_q = reinterpret_cast<producer_q*>((base_ptr + empty_frameset_producer_q_offset));

  return 0;
}

void cleanup_streams(struct stream_ctx& ctx) {
  if (ctx.mmap_buf != nullptr)
    munmap(ctx.mmap_buf, ctx.shm_size);

  if (ctx.shm_fd >= 0)
    close(ctx.shm_fd);

  if (ctx.server_pid > 0)
    kill(ctx.server_pid, SIGTERM);
}
