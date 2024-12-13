#ifndef LOCKFREE_CONTAINERS_H
#define LOCKFREE_CONTAINERS_H

#include <stdatomic.h>
#include <stdint.h>

struct lf_node;

struct next_ptr {
  struct lf_node* ptr;
  uint32_t count;
};

struct lf_node {
  _Atomic(struct next_ptr) next;
  void* data;
};

struct lf_stack {
  _Atomic(struct next_ptr) head;
};

void lf_stack_init(struct lf_stack* stack);
void lf_stack_push(struct lf_stack* stack, struct lf_node* node);
struct lf_node* lf_stack_pop(struct lf_stack* stack);

struct lf_queue {
  _Atomic(struct next_ptr) head;
  _Atomic(struct next_ptr) tail;
};

void lf_queue_init(struct lf_queue* queue, struct lf_node* dummy);
void lf_queue_nq(struct lf_queue* queue, struct lf_node* node);
struct lf_node* lf_queue_dq(struct lf_queue* queue);

#endif // LOCKFREE_CONTAINERS_H
