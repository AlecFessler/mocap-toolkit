#ifndef PROCESS_CONTEXT_H
#define PROCESS_CONTEXT_H

#include <atomic>
#include <cstdint>
#include <semaphore.h>
#include <signal.h>
#include "camera_handler.h"
#include "lock_free_queue.h"
#include "logger.h"

class camera_handler_t;

constexpr uint64_t IMAGE_WIDTH = 1920;
constexpr uint64_t IMAGE_HEIGHT = 1080;
constexpr uint64_t IMAGE_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT;
constexpr int PREALLOCATED_BUFFERS = 8;

class p_ctx_t {
public:
  logger_t* logger;
  camera_handler_t* cam;
  lock_free_queue_t* frame_queue;
  volatile sig_atomic_t running;
  sem_t* thread1_ready;
  sem_t* thread2_ready;
  sem_t* queue_counter;

  static p_ctx_t& get_instance() noexcept;

  p_ctx_t(p_ctx_t const&) = delete;
  void operator=(p_ctx_t const&) = delete;
  p_ctx_t(p_ctx_t&&) = delete;
  void operator=(p_ctx_t&&) = delete;

private:
  p_ctx_t() noexcept;
};

#endif // PROCESS_CONTEXT_H
