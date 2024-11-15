// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include "lock_free_node.h"
#include "lock_free_queue.h"
#include "lock_free_stack.h"

lock_free_queue_t::lock_free_queue_t(
  int prealloc_count
) :
  prealloc_nodes(new lock_free_node_t[prealloc_count]) {
  for (int i = 0; i < prealloc_count; i++)
    available_nodes.push(&prealloc_nodes[i]);
  lock_free_node_t* dummy = available_nodes.pop();
  dummy->next.store(nullptr, std::memory_order_relaxed);
  dummy->data = nullptr;
  head.store(dummy, std::memory_order_relaxed);
  tail.store(dummy, std::memory_order_relaxed);
}

lock_free_queue_t::~lock_free_queue_t() {
  if (prealloc_nodes) delete[] prealloc_nodes;
}

bool lock_free_queue_t::enqueue(void* data) noexcept {
  lock_free_node_t* new_node = available_nodes.pop();
  if (!new_node) return false;

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

  return true;
}

void* lock_free_queue_t::dequeue() noexcept {
  void* data = nullptr;

  while(true) {
    lock_free_node_t* head_node = head.load(std::memory_order_acquire);
    lock_free_node_t* tail_node = tail.load(std::memory_order_acquire);
    lock_free_node_t* next = head_node->next.load(std::memory_order_acquire);

    if (head_node == head.load(std::memory_order_acquire)) {
      if (head_node == tail_node) {
        if (next == nullptr) {
          return nullptr;
        }
        tail.compare_exchange_weak(tail_node, next, std::memory_order_release);
      } else {
        data = next->data;
        if (head.compare_exchange_weak(head_node, next, std::memory_order_release)) {
          available_nodes.push(head_node);
          break;
        }
      }
    }
  }

  return data;
}

bool lock_free_queue_t::empty() const noexcept {
  lock_free_node_t* head_node = head.load(std::memory_order_acquire);
  lock_free_node_t* tail_node = tail.load(std::memory_order_acquire);
  lock_free_node_t* next = head_node->next.load(std::memory_order_acquire);
  return head_node == tail_node && next == nullptr;
}
