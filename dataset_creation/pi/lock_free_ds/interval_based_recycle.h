#ifndef INTERVAL_BASED_RECYCLE_H
#define INTERVAL_BASED_RECYCLE_H

#include <atomic>
#include <pthread>
#include "lock_free_stack.h"

struct reservation_t {
  std::uint64_t lower;
  std::uint64_t upper;
};

struct memory_block_t {
  std::uint64_t created_epoch;
  std::uint64_t retired_epoch;
  void* data;
};

class interval_based_recycle_t {
public:
  interval_based_recycle_t(int num_threads, int prealloc_size, int epoch_freq, int recycle_freq);
  ~interval_based_recycle_t();

  int thread_idx();
  memory_block_t* get_block();
  void retire_block(memory_block_t* block);
  void start_op();
  void end_op();
  void* read_block(memory_block_t* block);

private:

  std::atomic<std::uint64_t> epoch;
  int num_threads_;
  reservation_t* reservations;

  stack_node_t* stack_nodes;
  memory_block_t* memory_blocks;

  int epoch_freq_;
  int recycle_freq_;
  static thread_local std::uint64_t counter;

  lock_free_stack_t available_blocks;
  lock_free_stack_t empty_stack_nodes;
  lock_free_stack_t retired_blocks;

  void recycle_blocks();
};

#endif // INTERVAL_BASED_RECYCLE_H
