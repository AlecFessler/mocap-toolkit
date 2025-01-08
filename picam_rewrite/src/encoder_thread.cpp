// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
extern "C" {
#include <libavcodec/avcodec.h>
}
#include <optional>
#include <pthread.h>
#include <thread>
#include <stdexcept>
#include <string>

#include "frame_buffer.hpp"
#include "encoder.hpp"
#include "encoder_thread.hpp"
#include "logging.hpp"
#include "packet_buffer.hpp"
#include "scheduling.hpp"
#include "sigsets.hpp"
#include "spsc_queue_wrapper.hpp"

constexpr std::chrono::microseconds SLEEP_DURATION{100};

static volatile sig_atomic_t stop_flag = 0;
static void stop_handler(int signum) {
  (void)signum;
  stop_flag = 1;
}

void* encoder_thread_fn(void* ptr) {
  auto instance = static_cast<EncoderThread*>(ptr);
  try{
    pin_to_core(1);
    set_scheduling_prio(99);
    setup_sig_handler(SIGTERM, stop_handler);
    while (!stop_flag) {
      std::optional<struct frame> frame = instance->m_frame_queue.try_dequeue();
      while (!frame.has_value() && !stop_flag) {
        std::this_thread::sleep_for(SLEEP_DURATION);
        frame = instance->m_frame_queue.try_dequeue();
      }
      if (stop_flag) break;

      instance->m_encoder.encode(
        frame.value().buffer,
        instance->m_avpackets[instance->m_next_buffer]
      );
      std::span<uint8_t> packet_buffer{
        instance->m_avpackets[instance->m_next_buffer]->data,
        static_cast<size_t>(instance->m_avpackets[instance->m_next_buffer]->size)
      };
      instance->m_packet_buffers[instance->m_next_buffer].buffer = packet_buffer;
      instance->m_packet_buffers[instance->m_next_buffer].timestamp = frame.value().timestamp;

      bool enqueued = instance->m_packet_queue.try_enqueue(
        instance->m_packet_buffers[instance->m_next_buffer]
      );
      while (!enqueued && !stop_flag) {
        std::this_thread::sleep_for(SLEEP_DURATION);
        enqueued = instance->m_packet_queue.try_enqueue(
          instance->m_packet_buffers[instance->m_next_buffer]
        );
      }
      if (stop_flag) break;

      if (++instance->m_next_buffer == instance->m_packet_buffers.capacity())
        instance->m_next_buffer = 0;
    }

    return nullptr;
  } catch (...) {
    // any exception will have already been logged, so we don't need to read the exception
    // but the system only throws for critical errors, so we do need to shutdown the system
    instance->m_main_stop_flag.store(1, std::memory_order_release);
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
  m_next_buffer(0),
  m_avpackets(num_packets),
  m_packet_buffers(num_packets),
  m_encoder(resolution, fps, m_avpackets),
  m_frame_queue(frame_queue),
  m_packet_queue(packet_queue),
  m_main_stop_flag(main_stop_flag) {}

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
