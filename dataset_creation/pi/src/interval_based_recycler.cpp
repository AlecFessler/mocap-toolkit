// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include "interval_based_recycler.h"

thread_local uint64_t interval_based_recycler_t::counter = 0;

interval_based_recycler_t::interval_based_recycler_t(
  int num_threads,
  int epoch_freq,
  int recycle_freq,
  int recycle_batch_size,
  int prealloc_count
) :
  epoch(0),
  epoch_freq(epoch_freq),
  recycle_freq(recycle_freq),
  recycle_batch_size(recycle_batch_size),
  num_threads(num_threads),
  reservation_enumerator(0),
  reservations(new reservation_t[num_threads]),
  prealloc_count(prealloc_count),
  prealloc_nodes(new lock_free_node_t[prealloc_count]) {
  for (int i = 0; i < num_threads; i++)
    reservations[i].lower = reservations[i].upper = std::numeric_limits<uint64_t>::max();
  for (int i = 0; i < prealloc_count; i++)
    available_nodes.push(&prealloc_nodes[i]);
}

interval_based_recycler_t::~interval_based_recycler_t() {
  if (prealloc_nodes) delete[] prealloc_nodes;
  if (reservations) delete[] reservations;
}

int interval_based_recycler_t::thread_idx() noexcept {
  // using more than num_threads is not supported
  // and will result in undefined behavior
  static thread_local int idx = reservation_enumerator.fetch_add(1, std::memory_order_release) % num_threads;
  return idx;
}

lock_free_node_t* interval_based_recycler_t::get_node() noexcept {
  if (available_nodes.empty())
    recycle_nodes();
  if (++counter % epoch_freq == 0)
    epoch.fetch_add(1, std::memory_order_relaxed);

  lock_free_node_t* node = available_nodes.pop();
  if (!node) return nullptr;

  uint64_t current_epoch = epoch.load(std::memory_order_acquire);
  node->created_epoch = current_epoch;
  node->next = nullptr;

  return node;
}

void interval_based_recycler_t::retire_node(lock_free_node_t* node) noexcept {
  if (++counter % recycle_freq == 0)
    recycle_nodes();

  uint64_t current_epoch = epoch.load(std::memory_order_acquire);
  node->retired_epoch = current_epoch;

  retired_nodes.push(node);
}

void interval_based_recycler_t::start_op() noexcept {
  int idx = thread_idx();
  uint64_t current_epoch = epoch.load(std::memory_order_acquire);
  reservations[idx].lower = current_epoch;
  reservations[idx].upper = current_epoch;
}

void interval_based_recycler_t::end_op() noexcept {
  int idx = thread_idx();
  reservations[idx].lower = std::numeric_limits<uint64_t>::max();
  reservations[idx].upper = std::numeric_limits<uint64_t>::max();
}

void* interval_based_recycler_t::read_node(lock_free_node_t* node) noexcept {
  int idx = thread_idx();
  reservations[idx].upper = node->created_epoch;
  return node->data;
}

void interval_based_recycler_t::recycle_nodes() noexcept {
  int reserved_idx = 0;
  lock_free_node_t* reserved_nodes[recycle_batch_size];

  while (!retired_nodes.empty() && reserved_idx < recycle_batch_size) {
    lock_free_node_t* node = retired_nodes.pop();
    if (!node) break;

    bool reserved = false;
    for (int i = 0; i < num_threads; i++) {
      if (node->created_epoch <= reservations[i].upper && node->retired_epoch >= reservations[i].lower) {
        reserved = true;
        break;
      }
    }

    if (reserved)
      reserved_nodes[reserved_idx++] = node;
    else
      available_nodes.push(node);
  }

  for (int i = 0; i < reserved_idx; i++)
    retired_nodes.push(reserved_nodes[i]);
}
