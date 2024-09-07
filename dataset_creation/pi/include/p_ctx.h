// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

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

constexpr unsigned int IMAGE_WIDTH = 1920;
constexpr unsigned int IMAGE_HEIGHT = 1080;
constexpr unsigned int Y_PLANE = IMAGE_WIDTH * IMAGE_HEIGHT;
constexpr unsigned int UV_PLANE = Y_PLANE / 4;
constexpr unsigned int IMAGE_BYTES = Y_PLANE + UV_PLANE * 2;
constexpr unsigned int PREALLOCATED_BUFFERS = 8;

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
