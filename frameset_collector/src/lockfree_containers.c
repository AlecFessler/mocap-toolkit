#include <errno.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>

#include "lockfree_containers.h"

static inline void atmc_store(
  _Atomic struct next_ptr *obj,
  struct next_ptr value,
  memory_order order
) {
  atomic_store_explicit(obj, value, order);
}

static inline struct next_ptr atmc_load(
  _Atomic struct next_ptr *obj,
  memory_order order
) {
  return atomic_load_explicit(obj, order);
}

static inline bool cas_weak(
  _Atomic struct next_ptr *obj,
  struct next_ptr *expected,
  struct next_ptr desired,
  memory_order success,
  memory_order failure
) {
  return atomic_compare_exchange_weak_explicit(obj, expected, desired, success, failure);
}

static inline bool cas_strong(
  _Atomic struct next_ptr *obj,
  struct next_ptr *expected,
  struct next_ptr desired,
  memory_order success,
  memory_order failure
) {
  return atomic_compare_exchange_strong_explicit(obj, expected, desired, success, failure);
}

void lf_stack_init(struct lf_stack* stack) {
  atmc_store(
    &stack->head,
    ((struct next_ptr){NULL, 0}),
    memory_order_relaxed
  );
}

void lf_stack_push(struct lf_stack* stack, struct lf_node* node) {
  // load the current head
  struct next_ptr old_head = atmc_load(
    &stack->head,
    memory_order_acquire
  );

  bool swapped = false;
  while (!swapped) {
    // store the current head as our node's next ptr
    atmc_store(
      &node->next,
      old_head,
      memory_order_relaxed
    );

    // try to set our node as the new head
    swapped = cas_weak(
      &stack->head,
      &old_head,
      ((struct next_ptr){node, old_head.count + 1}),
      memory_order_release,
      memory_order_acquire
    );
  }
}

struct lf_node* lf_stack_pop(struct lf_stack* stack) {
  // load the current head
  struct next_ptr old_head = atmc_load(
    &stack->head,
    memory_order_acquire
  );

  bool swapped = false;
  while (!swapped) {
    if (old_head.ptr == NULL)
      return NULL; // stack is empty

    // load the node after the head
    struct next_ptr next = atmc_load(
      &old_head.ptr->next,
      memory_order_acquire
    );

    // try to swap the head with the node after
    swapped = cas_weak(
      &stack->head,
      &old_head,
      ((struct next_ptr){next.ptr, old_head.count + 1}),
      memory_order_release,
      memory_order_acquire
    );
  }

  return old_head.ptr;
}

void lf_queue_init(struct lf_queue* queue, struct lf_node* nodes, size_t num_nodes) {
  // setup the dummy node at the first node in nodes
  atmc_store(
    &nodes->next,
    ((struct next_ptr){NULL, 0}),
    memory_order_relaxed
  );
  atmc_store(
    &queue->head,
    ((struct next_ptr){nodes, 0}),
    memory_order_relaxed
  );
  atmc_store(
    &queue->tail,
    ((struct next_ptr){nodes, 0}),
    memory_order_relaxed
  );

  // initialize the node allocator and push the remaining nodes
  lf_stack_init(&queue->node_allocator);
  for (size_t i = 1; i < num_nodes; i++) {
    lf_stack_push(
      &queue->node_allocator,
      &nodes[i]
    );
  }
}

int lf_queue_nq(struct lf_queue* queue, void* data) {
  // get a new node from the allocator
  struct lf_node* node = lf_stack_pop(&queue->node_allocator);
  if (node == NULL)
    return -EAGAIN;

  node->data = data;

  // store dummy node in our node's next
  atmc_store(
    &node->next,
    ((struct next_ptr){NULL, 0}),
    memory_order_release
  );

  bool swapped = false;
  while (!swapped) {
    // load the current tail
    struct next_ptr current_tail = atmc_load(
      &queue->tail,
      memory_order_acquire
    );

    // load the tail's next (expected NULL if tail is up to date)
    struct next_ptr tail_next = atmc_load(
      &current_tail.ptr->next,
      memory_order_acquire
    );

    // load the tail again to verify it hasn't changed
    struct next_ptr tail_check = atmc_load(
      &queue->tail,
      memory_order_acquire
    );

    if (current_tail.ptr != tail_check.ptr ||
        current_tail.count != tail_check.count)
      continue; // tail has changed, try again

    // tail is falling behind, try to correct it
    if (tail_next.ptr != NULL) {
      cas_strong(
        &queue->tail,
        &current_tail,
        ((struct next_ptr){tail_next.ptr, current_tail.count + 1}),
        memory_order_release,
        memory_order_relaxed
      );
      continue;
    }

    // try to swap our node into the tails next ptr
    swapped = cas_weak(
      &current_tail.ptr->next,
      &tail_next,
      ((struct next_ptr){node, current_tail.count + 1}),
      memory_order_release,
      memory_order_relaxed
    );

    // succeeded in linking new tail
    // try to update tail ptr
    // it's okay to return if it fails
    if (swapped)
      cas_strong(
        &queue->tail,
        &current_tail,
        ((struct next_ptr){node, current_tail.count + 1}),
        memory_order_release,
        memory_order_relaxed
      );
  }

  return 0;
}

void* lf_queue_dq(struct lf_queue* queue) {
  void* data = NULL;
  bool swapped = false;
  while (!swapped) {
    // load the head
    struct next_ptr head = atmc_load(
      &queue->head,
      memory_order_acquire
    );

    // load the tail
    struct next_ptr tail = atmc_load(
      &queue->tail,
      memory_order_acquire
    );

    // load the head's next
    struct next_ptr head_next = atmc_load(
      &head.ptr->next,
      memory_order_acquire
    );

    // load the head again to ensure it's unchanged
    struct next_ptr head_check = atmc_load(
      &queue->head,
      memory_order_acquire
    );

    if (head.ptr != head_check.ptr ||
        head.count != head_check.count)
      continue; // head has changed, try again

    if (head.ptr == tail.ptr) {
      if (head_next.ptr == NULL)
        return NULL; // queue is empty

      // tail is falling behind, try to correct it
      cas_strong(
        &queue->tail,
        &tail,
        ((struct next_ptr){head_next.ptr, tail.count + 1}),
        memory_order_release,
        memory_order_relaxed
      );
      continue;
    }

    // get the data ptr from heads next
    data = head_next.ptr->data;

    // try to swap the head with next to remove it
    swapped = cas_weak(
      &queue->head,
      &head,
      ((struct next_ptr){head_next.ptr, head.count + 1}),
      memory_order_release,
      memory_order_relaxed
    );

    if (swapped) // put the lf_node back in the allocator
      lf_stack_push(&queue->node_allocator, head.ptr);
  }

  return data;
}
