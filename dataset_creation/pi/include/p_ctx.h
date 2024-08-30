#ifndef PROCESS_CONTEXT_H
#define PROCESS_CONTEXT_H

#include <atomic>
#include <cstdint>
#include <semaphore.h>
#include "camera_handler.h"
#include "lock_free_queue.h"
#include "logger.h"
#include "mm_lock_free_stack.h"

class camera_handler_t;

constexpr uint64_t IMAGE_WIDTH = 1920;
constexpr uint64_t IMAGE_HEIGHT = 1080;
constexpr uint64_t IMAGE_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT;
constexpr int PREALLOCATED_BUFFERS = 32;

class p_ctx_t {
public:
  logger_t* logger;
  camera_handler_t* cam;
  mm_lock_free_stack_t* available_buffers;
  lock_free_queue_t* frame_queue;
  std::atomic<bool> running;
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
