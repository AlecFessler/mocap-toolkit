#include <sched.h>
#include <semaphore.h>
#include "stream_thread.h"
#include "p_ctx.h"

void* stream_thread(void* p_ctx) {
  p_ctx_t& ctx = *(p_ctx_t*)p_ctx;

  struct sched_param param;
  param.sched_priority = sched_get_priority_max(SCHED_FIFO) - 1;
  if (sched_setscheduler(0, SCHED_FIFO, &param) < 0) {
    const char* err = "Failed to set real-time scheduling policy";
    ctx.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    return nullptr;
  }

  int enqueued_frames = 0;
  sem_post(ctx.thread2_ready);
  sem_wait(ctx.thread1_ready);
  while (
    ctx.running.load(std::memory_order_acquire) ||
    enqueued_frames > 0
  ) {
    sem_wait(ctx.queue_counter);

    const char* info = "Dequeued frame";
    ctx.logger->log(logger_t::level_t::INFO, __FILE__, __LINE__, info);

    void* frame = ctx.frame_queue->dequeue();
    if (frame == nullptr) {
      const char* err = "Failed to dequeue frame";
      ctx.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
      return nullptr;
    }

    // send frame to tcp server

    bool pushed = false;
    do {
      pushed = ctx.available_buffers->push(frame);
    } while (!pushed);

    sem_getvalue(ctx.queue_counter, &enqueued_frames);
  }

  return nullptr;
}
