#ifndef LOCK_FREE_STACK_H
#define LOCK_FREE_STACK_H

#include <atomic>

struct stack_node_t {
  std::atomic<stack_node_t*> next;
  void* ptr;
};

class lock_free_stack_t {
public:
  lock_free_stack_t();
  // Stack is not responsible for allocating or freeing memory.
  // It only stores pointers to preallocated memory because it
  // is lock-free and allocating heap memory is not lock-free

  void push(stack_node_t* node);
  stack_node_t* pop();
  bool empty();

private:
  std::atomic<stack_node_t*> head;
};

#endif // LOCK_FREE_STACK_H
