// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef PROTOCOL_HPP
#define PROTOCOL_HPP

#include <cstdint>

// TCP control message types
enum ctrl_msg_type : uint8_t {
  CTRL_START     = 0x01,  // server -> pi: payload = [uint64_t timestamp]
  CTRL_STOP      = 0x02,  // server -> pi: no payload
  CTRL_ACK       = 0x03,  // pi -> server: no payload
  CTRL_HEARTBEAT = 0x04,  // bidirectional: no payload
  CTRL_IDENTIFY  = 0x05,  // pi -> server: payload = [uint8_t camera_id]
};

// TCP control message header
struct __attribute__((packed)) ctrl_msg_header {
  uint8_t type;
  uint32_t length; // payload length (0 for no payload)
};

constexpr uint32_t CTRL_HEADER_SIZE = sizeof(ctrl_msg_header);

// TCP video stream frame header
struct __attribute__((packed)) tcp_frame_header {
  uint32_t sequence;    // frame sequence number (monotonic per camera)
  uint64_t timestamp;   // capture timestamp (nanoseconds since epoch)
  uint32_t size;        // total payload size in bytes
};

constexpr uint32_t TCP_FRAME_HEADER_SIZE = sizeof(tcp_frame_header);

// Heartbeat timing
constexpr uint32_t HEARTBEAT_INTERVAL_MS = 1000;
constexpr uint32_t HEARTBEAT_TIMEOUT_MS = 3000;

#endif // PROTOCOL_HPP
