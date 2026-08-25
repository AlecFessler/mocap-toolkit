// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <optional>
#include <pthread.h>
#include <thread>
#include <stdexcept>
#include <string>

#include "logging.hpp"
#include "packet_buffer.hpp"
#include "scheduling.hpp"
#include "sigsets.hpp"
#include "spsc_queue_wrapper.hpp"
#include "stream_thread.hpp"
#include "tcp_socket.hpp"

constexpr std::chrono::microseconds SLEEP_DURATION{100};

void* stream_thread_fn(void* ptr) {
  auto instance = static_cast<StreamThread*>(ptr);
  try {
    pin_to_core(1);
    set_scheduling_prio(98);
    while (!instance->m_main_stop_flag.load(std::memory_order::acquire)) {
      std::optional<struct packet> packet = instance->m_packet_queue.try_dequeue();
      while (!packet.has_value() && !instance->m_main_stop_flag.load(std::memory_order::acquire)) {
        std::this_thread::sleep_for(SLEEP_DURATION);
        packet = instance->m_packet_queue.try_dequeue();
      }
      if (instance->m_main_stop_flag.load(std::memory_order::acquire))
        break;

      log_(BENCHMARK, "Started streaming packet");

      instance->m_tcpsock.stream_packet(
        packet.value().timestamp,
        packet.value().buffer
      );

      log_(BENCHMARK, "Finished streaming packet");
    }

    return nullptr;
  } catch (...) {
    // any exception will have already been logged, so we don't need to read the exception
    // but the system only throws for critical errors, so we do need to shutdown the system
    instance->m_main_stop_flag.store(1, std::memory_order_release);
    return nullptr;
  }
}

StreamThread::StreamThread(
  uint16_t port,
  std::string_view ip,
  SPSCQueue<struct packet>& packet_queue,
  std::atomic<bool>& main_stop_flag
) :
  m_tcpsock{port, ip},
  m_packet_queue(packet_queue),
  m_main_stop_flag(main_stop_flag) {}

StreamThread::~StreamThread() {
  pthread_join(m_this_thread, nullptr);
}

void StreamThread::launch() {
  int status = pthread_create(
    &m_this_thread,
    nullptr,
    stream_thread_fn,
    static_cast<void*>(this)
  );
  if (status != 0) {
    std::string err_msg =
      "Failed to spawn stream thread: "
      + std::string(strerror(errno));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }
}
