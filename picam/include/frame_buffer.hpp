#ifndef FRAME_BUFFER_HPP
#define FRAME_BUFFER_HPP

#include <chrono>
#include <cstdint>
#include <span>

struct frame {
  std::chrono::nanoseconds timestamp;
  std::span<uint8_t> buffer;
};

#endif // FRAME_BUFFER_HPP
