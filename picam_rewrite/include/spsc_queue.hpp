// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef SPSC_QUEUE_HPP
#define SPSC_QUEUE_HPP

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>

constexpr size_t CACHE_LINE_SIZE = 64;

struct producer_q {
  std::atomic<size_t> head;
  std::atomic<size_t>* tail_ptr;
  size_t cached_tail;
  size_t cap;
  void* buf;
  alignas(CACHE_LINE_SIZE) char padding[CACHE_LINE_SIZE];
};

struct consumer_q {
  std::atomic<size_t> tail;
  std::atomic<size_t>* head_ptr;
  size_t cached_head;
  size_t cap;
  void* buf;
  alignas(CACHE_LINE_SIZE) char padding[CACHE_LINE_SIZE];
};

static inline int spsc_queue_init(
  struct producer_q* pq,
  struct consumer_q* cq,
  void* buf,
  size_t size
) {
  pq->head.store(0, std::memory_order_relaxed);
  pq->tail_ptr = &cq->tail;
  pq->cached_tail = 0;
  pq->cap = size;
  pq->buf = buf;

  cq->tail.store(0, std::memory_order_relaxed);
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
  size_t head = q->head.load(std::memory_order_relaxed);

  size_t next = head + 1;
  if (next == q->cap)
    next = 0;

  if (next == q->cached_tail) { // check if the queue appears full
    q->cached_tail = q->tail_ptr->load(std::memory_order_acquire);
    if (next == q->cached_tail) // check if the queue actually is full
      return -EAGAIN;
  }

  void** slot = (void**)q->buf + head;
  *slot = data;

  // update the head with release semantics so the consumer
  // sees the assignment of the ptr to the slot before the
  // atomic store of the head index
  q->head.store(next, std::memory_order_release);

  return 0;
}

static inline void* spsc_dequeue(struct consumer_q* q) {
  // load tail with relaxed ordering, we're the only writer
  size_t tail = q->tail.load(std::memory_order_relaxed);

  if (tail == q->cached_head) { // check if the queue appears empty
    q->cached_head = q->head_ptr->load(std::memory_order_acquire);
    if (tail == q->cached_head) // check if the queue actually is empty
      return nullptr;
  }

  void** slot = (void**)q->buf + tail;
  void* data = *slot;

  size_t next_tail = tail + 1;
  if (next_tail == q->cap)
    next_tail = 0;

  // unlike in the enqueue, where we need the ptr assignment to
  // 'happen before' the atomic increment, there's no operation in
  // this function that needs 'happens before' ordering wrt the atomic store
  // so relaxed ordering is sufficient, the consumer thread will see the
  // correct data ptr since it was written to it's own cache
  q->tail.store(next_tail, std::memory_order_relaxed);

  return data;
}

#endif // SPSC_QUEUE_HPP
