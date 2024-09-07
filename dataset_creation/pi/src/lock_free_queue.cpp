// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include "interval_based_recycler.h"
#include "lock_free_queue.h"

lock_free_queue_t::lock_free_queue_t(
  int num_threads,
  int prealloc_count
) :
  recycler(
    num_threads,
    4, // epoch_freq
    4, // recycle_freq
    8, // recycle_batch_size
    prealloc_count
  ) {
  lock_free_node_t* dummy = recycler.get_node();
  dummy->next.store(nullptr, std::memory_order_relaxed);
  dummy->data = nullptr;
  head.store(dummy, std::memory_order_relaxed);
  tail.store(dummy, std::memory_order_relaxed);
}

bool lock_free_queue_t::enqueue(void* data) noexcept {
  recycler.start_op();

  lock_free_node_t* new_node = recycler.get_node();
  if (!new_node) {
    recycler.end_op();
    return false;
  }
  new_node->data = data;
  new_node->next.store(nullptr, std::memory_order_relaxed);

  while(true) {
    lock_free_node_t* current_tail = tail.load(std::memory_order_acquire);
    lock_free_node_t* next = current_tail->next.load(std::memory_order_acquire);

    if (current_tail == tail.load(std::memory_order_acquire)) {
      if (next == nullptr) {
        if (current_tail->next.compare_exchange_weak(next, new_node, std::memory_order_release)) {
          tail.compare_exchange_strong(current_tail, new_node, std::memory_order_release);
          break;
        }
      } else {
        tail.compare_exchange_weak(current_tail, next, std::memory_order_release);
      }
    }
  }

  recycler.end_op();
  return true;
}

void* lock_free_queue_t::dequeue() noexcept {
  recycler.start_op();
  void* data = nullptr;

  while(true) {
    lock_free_node_t* head_node = head.load(std::memory_order_acquire);
    lock_free_node_t* tail_node = tail.load(std::memory_order_acquire);
    lock_free_node_t* next = head_node->next.load(std::memory_order_acquire);

    if (head_node == head.load(std::memory_order_acquire)) {
      if (head_node == tail_node) {
        if (next == nullptr) {
          recycler.end_op();
          return nullptr;
        }
        tail.compare_exchange_weak(tail_node, next, std::memory_order_release);
      } else {
        data = next->data;
        if (head.compare_exchange_weak(head_node, next, std::memory_order_release)) {
          recycler.retire_node(head_node);
          break;
        }
      }
    }
  }

  recycler.end_op();
  return data;
}

bool lock_free_queue_t::empty() const noexcept {
  lock_free_node_t* head_node = head.load(std::memory_order_acquire);
  lock_free_node_t* tail_node = tail.load(std::memory_order_acquire);
  lock_free_node_t* next = head_node->next.load(std::memory_order_acquire);
  return head_node == tail_node && next == nullptr;
}
