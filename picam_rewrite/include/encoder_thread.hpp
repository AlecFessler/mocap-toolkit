// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef ENCODER_THREAD_HPP
#define ENCODER_THREAD_HPP

#include <atomic>
extern "C" {
#include <libavcodec/avcodec.h>
}
#include <pthread.h>
#include <vector>

#include "frame_buffer.hpp"
#include "encoder.hpp"
#include "packet_buffer.hpp"
#include "spsc_queue_wrapper.hpp"

class EncoderThread {
private:
  Encoder m_encoder;
  SPSCQueue<struct frame>& m_frame_queue;
  SPSCQueue<struct packet>& m_packet_queue;

  uint32_t m_next_buffer;
  std::vector<AVPacket*> m_avpackets;
  std::vector<struct packet> m_packet_buffers;

  pthread_t m_this_thread;
  std::atomic<bool>& m_main_stop_flag;

public:
  EncoderThread(
    std::pair<uint64_t, uint64_t> resolution,
    uint32_t fps,
    uint32_t num_packets,
    SPSCQueue<struct frame>& frame_queue,
    SPSCQueue<struct packet>& packet_queue,
    std::atomic<bool>& main_stop_flag
  );
  ~EncoderThread();
  void launch();
};

#endif // ENCODER_THREAD_HPP
