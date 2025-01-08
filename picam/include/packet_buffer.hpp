#ifndef PACKET_BUFFER_HPP
#define PACKET_BUFFER_HPP

#include <chrono>
#include <cstdint>
#include <span>

struct packet {
  std::chrono::nanoseconds timestamp;
  std::span<uint8_t> buffer;
};

#endif // PACKET_BUFFER_HPP
