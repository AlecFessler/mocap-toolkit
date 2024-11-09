// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef LOCK_FREE_QUEUE_H
#define LOCK_FREE_QUEUE_H

#include <atomic>
#include "interval_based_recycler.h"
#include "lock_free_node.h"

class lock_free_queue_t {
public:
  lock_free_queue_t(
    int num_threads,
    int prealloc_count
  );
  // recycler manages memory so no destructor needed

  bool enqueue(void* data) noexcept;
  void* dequeue() noexcept;
  bool empty() const noexcept;

private:
  std::atomic<lock_free_node_t*> head;
  std::atomic<lock_free_node_t*> tail;
  interval_based_recycler_t recycler;
};

#endif // LOCK_FREE_QUEUE_H
