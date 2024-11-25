// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef LOCK_FREE_QUEUE_H
#define LOCK_FREE_QUEUE_H

#include <atomic>
#include "lock_free_node.h"
#include "lock_free_stack.h"

class lock_free_queue_t {
public:
  lock_free_queue_t(
    int prealloc_count
  );
  ~lock_free_queue_t();

  bool enqueue(void* data) noexcept;
  void* dequeue() noexcept;
  bool empty() const noexcept;

private:
  std::atomic<lock_free_node_t::next_ptr_t> head;
  std::atomic<lock_free_node_t::next_ptr_t> tail;
  lock_free_node_t* prealloc_nodes;
  lock_free_stack_t available_nodes;
};

#endif // LOCK_FREE_QUEUE_H
