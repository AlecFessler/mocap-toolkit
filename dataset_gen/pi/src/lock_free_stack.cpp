// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include "lock_free_stack.h"

lock_free_stack_t::lock_free_stack_t() noexcept : head({nullptr, 0}) {}

void lock_free_stack_t::push(lock_free_node_t* node) noexcept {
  lock_free_node_t::next_ptr_t old_head;
  do {
    old_head = head.load(std::memory_order_acquire);
    node->next.store(
      {old_head.ptr, old_head.count + 1},
      std::memory_order_release
    );
  } while(!head.compare_exchange_weak(
      old_head,
      {node, old_head.count + 1},
      std::memory_order_release
    ));
}

lock_free_node_t* lock_free_stack_t::pop() noexcept {
  lock_free_node_t::next_ptr_t old_head;
  lock_free_node_t::next_ptr_t next;
  do {
    old_head = head.load(std::memory_order_acquire);
    if (old_head.ptr == nullptr)
      return nullptr;
    next = old_head.ptr->next.load(std::memory_order_acquire);
  } while (!head.compare_exchange_weak(
      old_head,
      {next.ptr, old_head.count + 1},
      std::memory_order_release
    ));
  return old_head.ptr;
}

bool lock_free_stack_t::empty() const noexcept {
  return head.load(std::memory_order_acquire).ptr == nullptr;
}
