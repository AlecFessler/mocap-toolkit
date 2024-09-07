// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef INTERVAL_BASED_RECYCLER_H
#define INTERVAL_BASED_RECYCLER_H

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include "lock_free_node.h"
#include "lock_free_stack.h"

struct reservation_t {
  uint64_t lower;
  uint64_t upper;
};

class interval_based_recycler_t {
public:
  interval_based_recycler_t(
    int num_threads,
    int epoch_freq,
    int recycle_freq,
    int recycle_batch_size,
    int prealloc_count
  );
  ~interval_based_recycler_t();

  int thread_idx() noexcept;
  lock_free_node_t* get_node() noexcept;
  void retire_node(lock_free_node_t* node) noexcept;
  void start_op() noexcept;
  void end_op() noexcept;
  void* read_node(lock_free_node_t* node) noexcept;

private:
  std::atomic<uint64_t> epoch;

  static thread_local uint64_t counter;
  int epoch_freq;
  int recycle_freq;
  int recycle_batch_size;

  int num_threads;
  std::atomic<int> reservation_enumerator;
  reservation_t* reservations;

  int prealloc_count;
  lock_free_node_t* prealloc_nodes;
  lock_free_stack_t available_nodes;
  lock_free_stack_t retired_nodes;

  void recycle_nodes() noexcept;
};

#endif // INTERVAL_BASED_RECYCLER_H
