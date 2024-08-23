#include "interval_based_recycle.h"
#include "lock_free_stack.h"

interval_based_recycle_t::interval_based_recycle_t(int num_threads, int prealloc_size, int epoch_freq, int recycle_freq) {
  epoch.store(0);

  num_threads_ = num_threads;
  reservations = new reservation_t[num_threads];
  for (int i = 0; i < num_threads; i++)
    reservations[i].lower = reservations[i].upper = std::numeric_limits<std::uint64_t>::max();

  epoch_freq_ = epoch_freq;
  recycle_freq_ = recycle_freq;

  stack_nodes = new stack_node_t[prealloc_size];
  memory_blocks = new memory_block_t[prealloc_size];
  for (int i = 0; i < prealloc_size; i++) {
    stack_nodes[i].ptr = (void*)&memory_blocks[i];
    available_blocks.push(&stack_nodes[i]);
  }
}

interval_based_recycle_t::~interval_based_recycle_t() {
  delete[] stack_nodes;
  delete[] memory_blocks;
  delete[] reservations;
}

int interval_based_recycle_t::thread_idx() {
  // using more than num_threads_ at once is not supported
  static std::atomic<std::uint64_t> global_thread_counter(0);
  static thread_local int thread_idx = global_thread_counter.fetch_add(1) % num_threads_;
  return thread_idx;
}

memory_block_t* interval_based_recycle_t::get_block() {
  if (available_blocks.empty())
    recycle_blocks();

  if (++counter % epoch_freq_ == 0) // thread local counter
    epoch.fetch_add(1, std::memory_order_relaxed);

  stack_node_t* node = available_blocks.pop();
  if (!node)
    // this should not happen unless,
    // there are not enough allocated blocks or,
    // this thread is under high contention
    return nullptr;

  memory_block_t* block = (memory_block_t*)node->ptr;
  node->ptr = nullptr;
  emtpy_stack_nodes.push(node);

  std::uint64_t current_epoch = epoch.load(std::memory_order_acquire);
  block->created_epoch = current_epoch;

  return block;
}

void interval_based_recycle_t::retire_block(memory_block_t* block) {
  if (++counter % recycle_freq_ == 0) // thread local counter
    recycle_blocks();

  stack_node_t* node = empty_stack_nodes.pop();
  if (!node)
    return;
  node->ptr = (void*)block;

  std::uint64_t current_epoch = epoch.load(std::memory_order_acquire);
  block->retired_epoch = current_epoch;

  retired_blocks.push(node);
}

void interval_based_recycle_t::start_op() {
  int tidx = thread_idx();
  std::uint64_t current_epoch = epoch.load(std::memory_order_acquire);
  reservations[tidx].lower = current_epoch;
  reservations[tidx].upper = current_epoch;
}

void interval_based_recycle_t::end_op() {
  int tidx = thread_idx();
  std::uint64_t current_epoch = epoch.load(std::memory_order_acquire);
  reservations[tidx].lower = std::numeric_limits<std::uint64_t>::max();
  reservations[tidx].upper = std::numeric_limits<std::uint64_t>::max();
}

void* interval_based_recycle_t::read_block(memory_block_t* block) {
  int tidx = thread_idx();
  reservations[tidx].upper = block->created_epoch;
  void* data = block->data;
  return data;
}

void interval_based_recycle_t::recycle_blocks() {
  constexpr int RESERVED_MAX = 64;
  int reserved_idx = 0;
  stack_node_t reserved_blocks[RESERVED_MAX];

  while(!retired_blocks.empty() && reserved_idx <= RESERVED_MAX - 1) {
    stack_node_t* node = retired_blocks.pop();
    if (!node)
      break;
    memory_block_t* block = (memory_block_t*)node->ptr;

    bool reserved = false;
    for (int i = 0; i < num_threads_; i++) {
      if (block->created_epoch <= reservations[i].upper && block->retired_epoch >= reservations[i].lower) {
        reserved = true;
        break;
      }
    }

    if (reserved)
      reserved_blocks[reserved_idx++] = node;
    else
      available_blocks.push(node);
  }

  for (int i = 0; i < reserved_idx; i++)
    retired_blocks.push(&reserved_blocks[i]);
}
