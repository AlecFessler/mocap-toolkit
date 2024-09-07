// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef LOCK_FREE_NODE_H
#define LOCK_FREE_NODE_H

#include <atomic>
#include <cstdint>

struct lock_free_node_t {
  uint64_t created_epoch;
  uint64_t retired_epoch;
  std::atomic<lock_free_node_t*> next;
  void* data;
};

#endif // LOCK_FREE_NODE_H
