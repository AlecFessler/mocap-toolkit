#ifndef LOCK_FREE_QUEUE_H
#define LOCK_FREE_QUEUE_H

#include <atomic>
#include "interval_based_recycler.h"

struct queue_node_t {
  std::atomic<memory_block_t<queue_node_t>*> next;
  void* data;
};

class lock_free_queue_t {
public:
  lock_free_queue_t(int num_threads, int prealloc_size);
  // no destructor because recycler cleans up

  bool enqueue(void* data);
  void* dequeue();
  bool empty();

private:
  std::atomic<memory_block_t<queue_node_t>*> head;
  std::atomic<memory_block_t<queue_node_t>*> tail;

  interval_based_recycler_t<queue_node_t> recycler;
};

#endif // LOCK_FREE_QUEUE_H
