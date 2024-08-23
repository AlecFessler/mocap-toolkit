#include "interval_based_recycler.h"
#include "lock_free_queue.h"

lock_free_queue_t::lock_free_queue_t(int num_threads, int prealloc_size) :
  // recycler magic numbers are:
  // epoch freq, recycle freq, recycle batch size
  recycler(num_threads, prealloc_size, 5, 5, 64) {

  memory_block_t<queue_node_t>* dummy_node = recycler.get_block();
  queue_node_t* dummy_node_ptr = dummy_node->data;
  dummy_node_ptr->next.store(nullptr, std::memory_order_release);
  dummy_node_ptr->data = nullptr;
  head.store(dummy_node, std::memory_order_relaxed);
  tail.store(dummy_node, std::memory_order_relaxed);
}

bool lock_free_queue_t::enqueue(void* data) {
  recycler.start_op();

  memory_block_t<queue_node_t>* new_block = recycler.get_block();
  if (!new_block) {
    recycler.end_op();
    return false;
  }

  queue_node_t* new_node = new_block->data;
  new_node->data = data;
  new_node->next.store(nullptr, std::memory_order_release);

  while(true) {
    memory_block_t<queue_node_t>* tail_block = tail.load(std::memory_order_acquire);
    queue_node_t* tail_ptr = tail_block->data;
    memory_block_t<queue_node_t>* tail_next_block = tail_ptr->next.load(std::memory_order_acquire);

    if (tail_block == tail.load(std::memory_order_acquire)) { // is tail still tail?
      if (tail_next_block != nullptr) { // tail is falling behind, try to advance it
        tail.compare_exchange_weak(tail_block, tail_next_block, std::memory_order_acq_rel);
        continue; // try again with hopefully updated tail
      // tail is up to date, try to enqueue new node
      } else if (tail_ptr->next.compare_exchange_weak(tail_next_block, new_block, std::memory_order_acq_rel)) {
        // try to update tail
        tail.compare_exchange_strong(tail_block, new_block, std::memory_order_acq_rel);
        break; // enqueued successfully
      }
    }
  }

  recycler.end_op();
  return true;
}

void* lock_free_queue_t::dequeue() {
  recycler.start_op();

  memory_block_t<queue_node_t>* head_block;
  void* data = nullptr;

  while(true) {
    head_block = head.load(std::memory_order_acquire);
    memory_block_t<queue_node_t>* tail_block = tail.load(std::memory_order_acquire);
    queue_node_t* head_ptr = head_block->data;
    memory_block_t<queue_node_t>* head_next_block = head_ptr->next.load(std::memory_order_acquire);

    if (head_block == head.load(std::memory_order_acquire)) { // is head still head?
      if (head_block == tail_block) { // is queue empty or tail falling behind?
        if (head_next_block == nullptr) // queue is empty
          break;
        // tail is falling behind, try to advance it
        tail.compare_exchange_weak(tail_block, head_next_block, std::memory_order_acq_rel);
      } else {
        // try to dequeue
        data = head_next_block->data->data;
        if (head.compare_exchange_weak(head_block, head_next_block, std::memory_order_acq_rel))
          break; // dequeued successfully
      }
    }
  }

  recycler.end_op();
  recycler.retire_block(head_block);
  return data;
}

bool lock_free_queue_t::empty() {
  memory_block_t<queue_node_t>* head_block = head.load(std::memory_order_acquire);
  memory_block_t<queue_node_t>* tail_block = tail.load(std::memory_order_acquire);
  queue_node_t* head_ptr = head_block->data;
  memory_block_t<queue_node_t>* head_next_block = head_ptr->next.load(std::memory_order_acquire);
  return head_block->data == tail_block->data && head_next_block->data == nullptr;
}
