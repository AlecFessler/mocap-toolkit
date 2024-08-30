#ifndef MM_LOCK_FREE_STACK_H
#define MM_LOCK_FREE_STACK_H

#include <atomic>
#include <interval_based_recycler.h>
#include "lock_free_node.h"

class mm_lock_free_stack_t {
public:
  mm_lock_free_stack_t(
    int num_threads,
    int prealloc_count
  );
  // recycler manages memory so no destructor needed

  bool push(void* data) noexcept;
  void* pop() noexcept;
  bool empty() const noexcept;

private:
  std::atomic<lock_free_node_t*> head;
  interval_based_recycler_t recycler;
};

#endif // MM_LOCK_FREE_STACK_H
