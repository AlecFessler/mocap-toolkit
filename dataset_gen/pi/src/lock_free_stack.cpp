// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include "lock_free_stack.h"

lock_free_stack_t::lock_free_stack_t() noexcept : head(nullptr) {}

void lock_free_stack_t::push(lock_free_node_t* node) noexcept {
  lock_free_node_t* old_head = head.load(std::memory_order_acquire);
  do {
    node->next.store(old_head, std::memory_order_release);
  } while(!head.compare_exchange_weak(old_head, node, std::memory_order_release));
}

lock_free_node_t* lock_free_stack_t::pop() noexcept {
  lock_free_node_t* old_head;
  lock_free_node_t* new_head;

  do {
    old_head = head.load(std::memory_order_acquire);
    if (old_head == nullptr)
      return nullptr;
    new_head = old_head->next.load(std::memory_order_acquire);
  } while (!head.compare_exchange_weak(old_head, new_head, std::memory_order_release));

  return old_head;
}

bool lock_free_stack_t::empty() const noexcept {
  return head.load(std::memory_order_acquire) == nullptr;
}
