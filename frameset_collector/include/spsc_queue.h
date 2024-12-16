#ifndef SPSC_QUEUE_H
#define SPSC_QUEUE_H

#include <stdatomic.h>
#include <stdalign.h>

#define CACHE_LINE_SIZE 64

/**
 * The buffer must be CACHE_LINE_SIZE aligned,
 * this is to prevent false sharing between threads,
 * which would cause cache coherency traffic that isn't
 * strictly necessary.
 *
 * The buffer size must be a power of two, this is because
 * the implementation uses the power of two bitwise AND
 * optimization to handle the index overflow around size.
 *
 * The actual capacity of the queue is size - 1, because
 * the queue needs to maintain a single empty slot to
 * distinguish between queue full (head + 1 == tail)
 * and queue empty (head == tail) states.
 */

struct spsc_queue {
  alignas(CACHE_LINE_SIZE) _Atomic size_t head;
  alignas(CACHE_LINE_SIZE) _Atomic size_t tail;
  alignas(CACHE_LINE_SIZE) size_t cached_head;
  alignas(CACHE_LINE_SIZE) size_t cached_tail;
  alignas(CACHE_LINE_SIZE) void* buf;
  size_t mask;
};

int spsc_queue_init(
  struct spsc_queue* q,
  void* buf,
  size_t size
);
int spsc_enqueue(
  struct spsc_queue* q,
  void* data
);
void* spsc_dequeue(struct spsc_queue* q);

#endif // SPSC_QUEUE_H
