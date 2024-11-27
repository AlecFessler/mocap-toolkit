// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include "lock_free_node.h"
#include "lock_free_queue.h"
#include "lock_free_stack.h"

lock_free_queue_t::lock_free_queue_t(
  /**
   * Creates a lock-free FIFO queue with preallocated nodes.
   *
   * The queue is implemented using the Michael-Scott algorithm, which maintains
   * a linked list of nodes with atomic head and tail pointers. A key feature
   * is the use of a dummy node to simplify edge cases in the dequeue operation.
   *
   * The initialization sequence:
   * 1. Preallocates all nodes up front to avoid dynamic allocation during operation
   * 2. Pushes nodes to an available nodes stack for reuse
   * 3. Creates a dummy node as the initial head/tail
   * 4. Sets up atomic pointers with ABA protection via counter
   *
   * Memory ordering is relaxed during initialization since no other threads
   * can access the queue yet.
   *
   * Parameters:
   *   prealloc_count: Number of nodes to preallocate, determines maximum queue size
   *
   * Note: The queue can hold prealloc_count-1 items since one node is used as dummy
   */
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
  /**
   * Cleans up the preallocated lock free nodes
   */
  if (prealloc_nodes) delete[] prealloc_nodes;
}

bool lock_free_queue_t::enqueue(void* data) noexcept {
  /**
   * Adds an item to the tail of the queue without locks.
   *
   * The enqueue operation follows these steps:
   * 1. Gets a free node from the available nodes stack
   * 2. Prepares the new node with its data and null next pointer
   * 3. Attempts to append it after the current tail:
   *    - If tail is current, try to link new node
   *    - If tail is behind, help advance it and retry
   *    - If tail changed, retry entire operation
   *
   * Memory ordering ensures:
   * - acquire: Sees latest tail and next pointers
   * - release: Makes node updates visible to other threads
   *
   * Parameters:
   *   data: Pointer to data to enqueue
   *
   * Returns:
   *   true if enqueue succeeded
   *   false if no nodes available (queue full)
   *
   * Note: The operation is lock-free but not wait-free - threads may need
   * to help other threads complete their operations before succeeding
   */
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
  /**
   * Removes and returns the item at the head of the queue without locks.
   *
   * The dequeue operation handles these cases:
   * 1. Empty queue: Returns null if head equals tail and tail's next is null
   * 2. Single item: Updates both head and tail
   * 3. Multiple items: Updates just the head
   *
   * The algorithm:
   * 1. Reads head, tail, and head's next pointer
   * 2. Verifies head hasn't changed (if it has, retry)
   * 3. Checks if queue is empty
   * 4. Attempts to advance head to next node
   * 5. Returns data and recycles old head node
   *
   * Memory ordering ensures:
   * - acquire: Sees latest head/tail/next pointers
   * - release: Makes head updates visible to other threads
   *
   * Returns:
   *   Pointer to dequeued data, or nullptr if queue is empty
   */
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
  /**
   * Checks if the queue is empty using atomic operations.
   *
   * A queue is empty when:
   * 1. Head and tail point to the same node (the dummy)
   * 2. That node's next pointer is null
   *
   * Memory ordering is acquire to ensure we see the latest
   * state of all pointers for a consistent check.
   *
   * Note: In a concurrent system, the queue's state may
   * change immediately after this check returns. The result
   * is only guaranteed for the instant the check occurred.
   *
   * Returns:
   *   true if the queue was empty when checked
   */
  lock_free_node_t::next_ptr_t head_node = head.load(std::memory_order_acquire);
  lock_free_node_t::next_ptr_t tail_node = tail.load(std::memory_order_acquire);
  lock_free_node_t::next_ptr_t next = head_node.ptr->next.load(std::memory_order_acquire);
  return head_node.ptr == tail_node.ptr && next.ptr == nullptr;
}
