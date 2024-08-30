#include "mm_lock_free_stack.h"

mm_lock_free_stack_t::mm_lock_free_stack_t(
  int num_threads,
  int prealloc_count
) :
  head(nullptr),
  recycler(
    num_threads,
    4, // epoch_freq
    4, // recycle_freq
    8, // recycle_batch_size
    prealloc_count
  ) {}

bool mm_lock_free_stack_t::push(void* data) noexcept {
  recycler.start_op();

  lock_free_node_t* new_node = recycler.get_node();
  if (!new_node) {
    recycler.end_op();
    return false;
  }
  new_node->data = data;

  lock_free_node_t* old_head = head.load(std::memory_order_acquire);
  do {
    new_node->next.store(old_head, std::memory_order_release);
  } while(!head.compare_exchange_weak(old_head, new_node, std::memory_order_acq_rel));

  recycler.end_op();
  return true;
}

void* mm_lock_free_stack_t::pop() noexcept {
  recycler.start_op();

  lock_free_node_t* old_head;
  do {
    old_head = head.load(std::memory_order_acquire);
    if (old_head == nullptr) {
      recycler.end_op();
      return nullptr;
    }
    lock_free_node_t* new_head = old_head->next.load(std::memory_order_acquire);
    if (head.compare_exchange_weak(old_head, new_head, std::memory_order_acq_rel)) {
      void* data = recycler.read_node(old_head);
      recycler.retire_node(old_head);
      recycler.end_op();
      return data;
    }
  } while (true);
}

bool mm_lock_free_stack_t::empty() const noexcept {
  return head.load(std::memory_order_acquire) == nullptr;
}
