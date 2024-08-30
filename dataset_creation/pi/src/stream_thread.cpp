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

    void* frame = ctx.frame_queue->dequeue();
    // frame will only be nullptr here during shutdown
    // because the parent thread only ever posts to the
    // semaphore if it has enqueued a frame, or if it is
    // shutting down, which allows us to break out of
    // this loop and return from the thread
    if (frame == nullptr)
      return nullptr;

    const char* info = "Dequeued frame";
    ctx.logger->log(logger_t::level_t::INFO, __FILE__, __LINE__, info);

    // send frame to tcp server

    sem_getvalue(ctx.queue_counter, &enqueued_frames);
  }

  return nullptr;
}
