#include "lock_free_stack.h"

lock_free_stack_t::lock_free_stack_t() : head(nullptr) {}

void lock_free_stack_t::push(stack_node_t* node) {
  stack_node_t* old_head = head.load(std::memory_order_acquire);
  do {
    node->next.store(old_head, std::memory_order_relaxed);
  } while(!head.compare_exchange_weak(old_head, node, std::memory_order_acq_rel));
}

stack_node_t* lock_free_stack_t::pop() {
  stack_node_t* old_head;
  stack_node_t* new_head;

  do {
    old_head = head.load(std::memory_order_acquire);
    if (old_head == nullptr)
      return nullptr;
    new_head = old_head->next.load(std::memory_order_relaxed);
  } while (!head.compare_exchange_weak(old_head, new_head, std::memory_order_acq_rel));

  return old_head;
}

bool lock_free_stack_t::empty() {
  return head.load(std::memory_order_acquire) == nullptr;
}
