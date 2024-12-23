#ifndef SPSC_QUEUE_H
#define SPSC_QUEUE_H

#include <errno.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdalign.h>

#define CACHE_LINE_SIZE 64

/**
 * This implementation uses two structs, one
 * for the producer, and one for the consumer.
 * The reasoning for this is because it prevents
 * false sharing between the threads. However,
 * naively implementing a single struct with padding
 * puts other key info, like the capacity or buffer
 * ptr in a separate cache line. Benchmarking demonstrated
 * that this absolutely crushed IPC (instructions per cycle)
 * by requiring multiple cache lines to complete an operation.
 * The dual struct approach provides the best of both worlds,
 * prevents false sharing, but also keeps everything needed
 * by a single thread in a single cache line. The only real
 * cost is a ptr dereference to read the other threads data,
 * but the head/tail caching optimization also limits the
 * frequency of this happening as well.
 *
 * The caching of the head and tail ptrs significantly
 * cuts down on cache coherency traffic. I found this
 * optimization from Erik Rigtorp's spsc queue implementation
 * on github, linked here: https://github.com/rigtorp/SPSCQueue
 *
 * The split struct optimization was my own doing. You could
 * hypothetically achieve a similar outcome with a single
 * struct or a C++ class, simply by using a redundant
 * copy of the capacity and buffer ptr, and splitting it
 * up with padding to prevent false sharing, while still
 * keeping each thread's 'workspace' in a single cache line
 */

struct producer_q {
  _Atomic size_t head;
  _Atomic size_t* tail_ptr;
  size_t cached_tail;
  size_t cap;
  void* buf;
  alignas(CACHE_LINE_SIZE) char padding[
    CACHE_LINE_SIZE
    - (sizeof(_Atomic size_t)
    + sizeof(_Atomic size_t*)
    + sizeof(size_t) * 2
    + sizeof(void*))
  ];
};

struct consumer_q {
  _Atomic size_t tail;
  _Atomic size_t* head_ptr;
  size_t cached_head;
  size_t cap;
  void* buf;
  alignas(CACHE_LINE_SIZE) char padding[
    CACHE_LINE_SIZE
    - (sizeof(_Atomic size_t)
    + sizeof(_Atomic size_t*)
    + sizeof(size_t) * 2
    + sizeof(void*))
  ];
};

static inline int spsc_queue_init(
  struct producer_q* pq,
  struct consumer_q* cq,
  void* buf,
  size_t size
) {
  atomic_store_explicit(&pq->head, 0, memory_order_relaxed);
  pq->tail_ptr = &cq->tail;
  pq->cached_tail = 0;
  pq->cap = size;
  pq->buf = buf;

  atomic_store_explicit(&cq->tail, 0, memory_order_relaxed);
  cq->head_ptr = &pq->head;
  cq->cached_head = 0;
  cq->cap = size;
  cq->buf = buf;

  return 0;
}

static inline int spsc_enqueue(
  struct producer_q* q,
  void* data
) {
  // load head with relaxed ordering, we're the only writer
  size_t head = atomic_load_explicit(&q->head, memory_order_relaxed);

  size_t next = head + 1;
  if (next == q->cap)
    next = 0;

  if (next == q->cached_tail) { // check if the queue appears full
    q->cached_tail = atomic_load_explicit(q->tail_ptr, memory_order_acquire);
    if (next == q->cached_tail) // check if the queue actually is full
      return -EAGAIN;
  }

  void** slot = (void**)q->buf + head;
  *slot = data;

  // update the head with release semantics so the consumer
  // sees the assignment of the ptr to the slot before the
  // atomic store of the head index
  atomic_store_explicit(&q->head, next, memory_order_release);

  return 0;
}

static inline void* spsc_dequeue(struct consumer_q* q) {
  // load tail with relaxed ordering, we're the only writer
  size_t tail = atomic_load_explicit(&q->tail, memory_order_relaxed);

  if (tail == q->cached_head) { // check if the queue appears empty
    q->cached_head = atomic_load_explicit(q->head_ptr, memory_order_acquire);
    if (tail == q->cached_head) // check if the queue actually is empty
      return NULL;
  }

  void** slot = (void**)q->buf + tail;
  void* data = *slot;

  size_t next_tail = tail + 1;
  if (next_tail == q->cap)
    next_tail = 0;

  // update the tail with relaxed semantics, there's nothing with
  // regards to ordering that the producer cares for in this operation
  // (ie. unlike in the enqueue, where we need the ptr assignment to
  // 'happen before' the atomic increment, there's no operation in
  // this function that needs 'happens before' ordering wrt the atomic store
  atomic_store_explicit(&q->tail, next_tail, memory_order_relaxed);

  return data;
}

#endif // SPSC_QUEUE_H
