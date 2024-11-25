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
  dummy->next.store({nullptr, 0}, std::memory_order_relaxed);
  dummy->data = nullptr;
  head.store({dummy, 0}, std::memory_order_relaxed);
  tail.store({dummy, 0}, std::memory_order_relaxed);
}

lock_free_queue_t::~lock_free_queue_t() {
  if (prealloc_nodes) delete[] prealloc_nodes;
}

bool lock_free_queue_t::enqueue(void* data) noexcept {
  lock_free_node_t* new_node = available_nodes.pop();
  if (!new_node) return false;

  new_node->data = data;
  new_node->next.store({nullptr, 0}, std::memory_order_relaxed);

  while(true) {
    lock_free_node_t::next_ptr_t current_tail = tail.load(std::memory_order_acquire);
    lock_free_node_t::next_ptr_t next = current_tail.ptr->next.load(std::memory_order_acquire);

    if (current_tail != tail.load(std::memory_order_acquire))
      continue; // tail has changed, try again

    if (next.ptr == nullptr &&
      current_tail.ptr->next.compare_exchange_weak(
        next,
        {new_node, next.count + 1},
        std::memory_order_release
    )) {
      // tail was up to date and we succeeded in linking in a new tail to next
      tail.compare_exchange_strong(
        current_tail,
        {new_node, current_tail.count + 1},
        std::memory_order_release
      );
      break;
    }
    // tail is falling behind, try to update the ptr and try again
    tail.compare_exchange_weak(
      current_tail,
      {next.ptr, current_tail.count + 1},
      std::memory_order_release
    );
  }
  return true;
}

void* lock_free_queue_t::dequeue() noexcept {
  void* data = nullptr;
  while(true) {
    lock_free_node_t::next_ptr_t head_node = head.load(std::memory_order_acquire);
    lock_free_node_t::next_ptr_t tail_node = tail.load(std::memory_order_acquire);
    lock_free_node_t::next_ptr_t next = head_node.ptr->next.load(std::memory_order_acquire);

    if (head_node != head.load(std::memory_order_acquire))
      continue; // head has changed, try again

    if (head_node.ptr == tail_node.ptr) {
      if (next.ptr == nullptr)
          return nullptr; // queue is empty if head == tail and next is nullptr
      // tail is falling behind, try to update the ptr and try again
      tail.compare_exchange_weak(
        tail_node,
        {next.ptr, tail_node.count + 1},
        std::memory_order_release
      );
      continue;
    }

    data = next.ptr->data;
    if (head.compare_exchange_weak(
      head_node,
      {next.ptr, head_node.count + 1},
      std::memory_order_release
    )) {
        // succeeded in unlinking the head, push the node to the pool and return the data
        available_nodes.push(head_node.ptr);
        break;
    }
  }
  return data;
}

bool lock_free_queue_t::empty() const noexcept {
  lock_free_node_t::next_ptr_t head_node = head.load(std::memory_order_acquire);
  lock_free_node_t::next_ptr_t tail_node = tail.load(std::memory_order_acquire);
  lock_free_node_t::next_ptr_t next = head_node.ptr->next.load(std::memory_order_acquire);
  return head_node.ptr == tail_node.ptr && next.ptr == nullptr;
}
