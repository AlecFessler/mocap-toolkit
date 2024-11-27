// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include "lock_free_stack.h"

lock_free_stack_t::lock_free_stack_t() noexcept : head({nullptr, 0})
/**
 * Creates an empty lock-free stack using atomic operations.
 *
 * The stack is implemented as a singly-linked list with an atomic head
 * pointer. The head pointer combines both the node address and a counter
 * to prevent the ABA problem in concurrent operations.
 *
 * The ABA problem is when:
 * 1. Thread A reads head pointing to node A
 * 2. Thread B pops node A and pushes node B
 * 3. Thread C pops node B and pushes node A
 * 4. Thread A continues, thinking head hasn't changed
 *
 * Our counter increments with every modification, making old values
 * of head distinguishable even if they point to the same node.
 *
 * Memory ordering is relaxed for initialization since no other threads
 * can access the stack yet.
 */
{}

void lock_free_stack_t::push(lock_free_node_t* node) noexcept {
  /**
   * Pushes a node onto the stack without using locks.
   *
   * The push operation works by:
   * 1. Reading the current head atomically
   * 2. Setting the new node's next pointer to current head
   * 3. Attempting to update head to point to new node
   * 4. Retrying if head was modified by another thread
   *
   * Memory ordering ensures:
   * - acquire: Sees the latest head value before updating
   * - release: Makes the new node visible to other threads
   *
   * The operation is lock-free but not wait-free - a thread may
   * need multiple attempts if there is contention, but system-wide
   * progress is guaranteed as some thread will always succeed.
   *
   * Parameters:
   *   node: Pointer to the node to push onto the stack
   *
   * Note: Node must not be null and should not be currently
   * in use by any other data structure
   */
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
  /**
   * Removes and returns the top node from the stack without locks.
   *
   * The pop operation:
   * 1. Reads head atomically
   * 2. Returns null if stack is empty
   * 3. Reads next pointer of head node
   * 4. Attempts to update head to skip the top node
   * 5. Retries if head was modified by another thread
   *
   * Memory ordering ensures:
   * - acquire: Sees latest head and next pointers
   * - release: Makes head update visible to other threads
   *
   * Like push, this operation is lock-free but not wait-free.
   *
   * Returns:
   *   Pointer to the popped node, or nullptr if stack was empty
   *
   * Note: The caller becomes responsible for the returned node's
   * memory management
   */
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
  /**
   * Checks if the stack is empty using atomic operations.
   *
   * The check is straightforward - if head is null, the stack
   * is empty. Memory ordering is acquire to ensure we see the
   * latest state of the head pointer.
   *
   * Returns:
   *   true if the stack was empty when checked
   *
   * Note: In a concurrent system, the stack's state may change
   * immediately after this check returns. The result is only
   * guaranteed for the instant the check occurred.
   */
  return head.load(std::memory_order_acquire).ptr == nullptr;
}
