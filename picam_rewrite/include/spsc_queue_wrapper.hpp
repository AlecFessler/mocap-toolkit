// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef SPSC_QUEUE_WRAPPER_HPP
#define SPSC_QUEUE_WRAPPER_HPP

#include <cerrno>
#include <cstdint>
#include <vector>

#include "spsc_queue.hpp"

template <typename T>
class SPSCQueue {
private:
  struct producer_q pq;
  struct consumer_q cq;
  std::vector<T*> buffer;

public:
  SPSCQueue(uint64_t capacity) {
    buffer.resize(capacity);
    spsc_queue_init(
      &pq,
      &cq,
      static_cast<void*>(buffer.data()),
      capacity
    );
  }

  bool try_enqueue(const T& item) {
    int status = spsc_enqueue(
      &pq,
      static_cast<void*>(&item)
    );
    return status != -EAGAIN;
  }

  std::optional<T&> try_dequeue() {
    void* ptr = spsc_dequeue(&cq);
    if (ptr == nullptr)
      return std::nullopt;
    return *static_cast<T*>(ptr);
  }
};

#endif // SPSC_QUEUE_WRAPPER_HPP
