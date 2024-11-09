// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef LOCK_FREE_STACK_H
#define LOCK_FREE_STACK_H

#include <atomic>
#include "lock_free_node.h"

class lock_free_stack_t {
public:
  lock_free_stack_t() noexcept;
  // memory is managed by the allocator so no special destructor is needed

  void push(lock_free_node_t* node) noexcept;
  lock_free_node_t* pop() noexcept;
  bool empty() const noexcept;

private:
  std::atomic<lock_free_node_t*> head;
};

#endif // LOCK_FREE_STACK_H
