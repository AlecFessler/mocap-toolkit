// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <atomic>
#include <chrono>
#include <csignal>
extern "C" {
#include <libavcodec/avcodec.h>
}
#include <pthread.h>
#include <thread>

#include "frame_buffer.hpp"
#include "encoder.hpp"
#include "encoder_thread.hpp"
#include "logging.hpp"
#include "packet_buffer.hpp"
#include "sigsets.hpp"
#include "spsc_queue_wrapper.hpp"

constexpr std::chrono::microseconds SLEEP_DURATION{100};

static volatile sig_atomic_t stop_flag = 0;
static void stop_handler(int signum) {
  (void)signum;
  stop_flag = 1;
}

void* encoder_thread_fn(void* ptr) {
  auto this = static_cast<EncoderThread*>(ptr);
  try{
    setup_sig_handler(SIGTERM, stop_handler);
    while (!stop_flag) {
      std::optional<struct frame> frame = this->m_frame_queue.try_dequeue();
      while (!frame.has_value() && !stop_flag) {
        std::this_thread::sleep_for(SLEEP_DURATION);
        frame = this->m_frame_queue.try_dequeue();
      }
      if (stop_flag) break;

      this->m_encoder.encode(
        frame.buffer,
        this->m_avpackets[this->m_next_buffer]
      );
      std::span<uint8_t> packet_buffer{
        this->m_avpackets[this->m_next_buffer]->data,
        this->m_avpackets[this->m_next_buffer]->size
      };
      this->m_packet_buffers[this->m_next_buffer].buffer = packet_buffer;
      this->m_packet_buffers[this->m_next_buffer].timestamp = frame.timestamp;

      bool enqueued = this->m_packet_queue.try_enqueue(
        this->m_packet_buffers[this->m_next_buffer]
      );
      while (!enqueued && !stop_flag) {
        std::this_thread::sleep_for(SLEEP_DURATION);
        enqueued = this->m_packet_queue.try_enqueue(
          this->m_packet_buffers[this->m_next_buffer]
        );
      }
      if (stop_flag) break;

      if (++this->m_next_buffer == this->m_packet_buffers.capacity())
        this->m_next_buffer = 0;
    }

    return nullptr;
  } catch (...) {
    // any exception will have already been logged, so we don't need to read the exception
    // but the system only throws for critical errors, so we do need to shutdown the system
    this->m_main_stop_flag.store(1, std::memory_order_release);
    return nullptr;
  }
}

EncoderThread::EncoderThread(
  std::pair<uint64_t, uint64_t> resolution,
  uint32_t fps,
  uint32_t num_packets,
  SPSCQueue<struct frame>& frame_queue,
  SPSCQueue<struct packet>& packet_queue,
  std::atomic<bool>& main_stop_flag
) :
  m_frame_queue(frame_queue),
  m_packet_queue(packet_queue),
  m_next_buffer(0),
  m_main_stop_flag(main_stop_flag) {

  m_avpackets.resize(num_packets);
  m_packet_buffers.resize(num_packets);
  m_encoder{resolution, fps, m_avpackets};
}

EncoderThread::~EncoderThread() {
  pthread_kill(m_this_thread, SIGTERM);
  pthread_join(m_this_thread, nullptr);
  for (auto* packet : m_avpackets)
    av_packet_free(&packet);
}

void EncoderThread::launch() {
  int status = pthread_create(
    &m_this_thread,
    nullptr,
    encoder_thread_fn,
    static_cast<void*>(this)
  );
  if (status != 0) {
    std::string err_msg =
      "Failed to spawn encoder thread: "
      + std::string(strerror(errno));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }
}
