// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef LOCK_FREE_NODE_H
#define LOCK_FREE_NODE_H

#include <atomic>
#include <cstdint>

struct lock_free_node_t {
  struct alignas(8) next_ptr_t {
    lock_free_node_t* ptr;
    int32_t count;

    bool operator==(const next_ptr_t& other) const {
      return ptr == other.ptr && count == other.count;
    }

    bool operator!=(const next_ptr_t& other) const {
      return ptr != other.ptr || count != other.count;
    }
  };
  std::atomic<next_ptr_t> next;
  void* data;
};

#endif // LOCK_FREE_NODE_H
