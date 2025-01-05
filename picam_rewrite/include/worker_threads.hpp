// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef WORKER_THREADS_HPP
#define WORKER_THREADS_HPP

#include <array>
#include <chrono>
#include <memory>
#include <pthread.h>
#include <semaphore.h>
#include <span>

#include "encoder.hpp"
#include "tcp_socket.hpp"

constexpr uint32_t NUM_WORKERS = 2;

struct worker_ctx {
  pthread_t worker_tid;
  sem_t launch_task_sem;

  pthread_mutex_t* encoder_lock;
  Encoder* encoder;

  pthread_mutex_t* sock_lock;
  TcpSocket* tcpsock;

  std::chrono::nanoseconds timestamp;
  std::span<uint8_t> frame;
};

class WorkerThreads {
private:
  std::array<struct worker_ctx, NUM_WORKERS> m_worker_ctxs;
  uint32_t m_next_worker;

  pthread_mutex_t m_encoder_lock;
  Encoder m_encoder;

  pthread_mutex_t m_sock_lock;
  TcpSocket m_tcpsock;

public:
  WorkerThreads(
    Encoder&& encoder,
    TcpSocket&& tcpsock
  );
  WorkerThreads(WorkerThreads&& other) noexcept;
  ~WorkerThreads();
  void launch_workers();
  void start_task(
    std::chrono::nanoseconds timestamp,
    std::span<uint8_t> frame
  );
};

#endif // WORKER_THREADS_HPP
