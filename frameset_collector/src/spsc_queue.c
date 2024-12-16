#include <errno.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "spsc_queue.h"

int spsc_queue_init(
  struct spsc_queue* q,
  void* buf,
  size_t size
) {
  if ((uintptr_t)buf % CACHE_LINE_SIZE != 0)
    return -EINVAL;

  if (size == 0 || (size & (size - 1)) != 0)
    return -EINVAL;

  atomic_store_explicit(&q->head, 0, memory_order_relaxed);
  atomic_store_explicit(&q->tail, 0, memory_order_relaxed);
  q->cached_head = 0;
  q->cached_tail = 0;
  q->buf = buf;
  q->mask = size - 1;

  return 0;
}

int spsc_enqueue(
  struct spsc_queue* q,
  void* data
) {
  // load head with relaxed ordering, we're the only writer
  size_t head = atomic_load_explicit(&q->head, memory_order_relaxed);

  // always leave one extra slot in the queue
  // that way we can distinguish between empty (head == tail)
  // and full (head + 1 == tail) states
  size_t next = (head + 1) & q->mask;

  if (next == q->cached_tail) { // check if the queue appears full
    q->cached_tail = atomic_load_explicit(&q->tail, memory_order_acquire);
    if (next == q->cached_tail) // check if the queue actually is full
      return -EAGAIN;
  }

  void** slot = (void**)q->buf + head;
  *slot = data;

  // update the head with release semantics so the consumer can see it
  atomic_store_explicit(&q->head, next, memory_order_release);

  return 0;
}

void* spsc_dequeue(struct spsc_queue* q) {
  // load tail with relaxed ordering, we're the only writer
  size_t tail = atomic_load_explicit(&q->tail, memory_order_relaxed);

  if (tail == q->cached_head) { // check if the queue appears empty
    q->cached_head = atomic_load_explicit(&q->head, memory_order_acquire);
    if (tail == q->cached_head) // check if the queue actually is empty
      return NULL;
  }

  void** slot = (void**)q->buf + tail;
  void* data = *slot;

  size_t next_tail = (tail + 1) & q->mask;

  // update tail with relaxed semantics, the consumer (caller of this)
  // will see the correct value for data thanks to per thread sequential consistency
  // and it doesn't matter if the producer (enqueue caller) doesn't see the update
  // immediately, because this will only lead to an extra conservative "is full"
  // check, but will not put the queue into an invalid state
  atomic_store_explicit(&q->tail, next_tail, memory_order_relaxed);

  return data;
}
