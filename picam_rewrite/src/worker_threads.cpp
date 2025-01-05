// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <chrono>
#include <csignal>
#include <cstring>
#include <pthread.h>
#include <semaphore.h>
#include <stdexcept>
#include <string>

#include "logging.hpp"
#include "scheduling.hpp"
#include "sigsets.hpp"
#include "worker_threads.hpp"

constexpr int WORKER_STOP_SIG = SIGUSR2;
static volatile sig_atomic_t stop_flag = 0;
static void stop_handler(int signum) {
  (void)signum;
  stop_flag = 1;
}

void* worker_thread_fn(void* ptr) {
  auto ctx = static_cast<struct worker_ctx*>(ptr);
  setup_sig_handler(WORKER_STOP_SIG, stop_handler);

  pin_to_core(1);
  set_scheduling_prio(99);

  while (!stop_flag) {
    sem_wait(&ctx->launch_task_sem);
    if (stop_flag)
      break;

    pthread_mutex_lock(ctx->encoder_lock);
    std::span<uint8_t> pkt = ctx->encoder->encode(ctx->frame);
    pthread_mutex_unlock(ctx->encoder_lock);
    log_(BENCHMARK, "Encoding complete");

    pthread_mutex_lock(ctx->sock_lock);
    ctx->tcpsock->stream_pkt(ctx->timestamp, pkt);
    pthread_mutex_unlock(ctx->sock_lock);
    log_(BENCHMARK, "Streaming complete");
  }

  return nullptr;
}

WorkerThreads::WorkerThreads(
  Encoder&& encoder,
  TcpSocket&& tcpsock
) :
  m_next_worker(0),
  m_encoder(std::move(encoder)),
  m_tcpsock(std::move(tcpsock)) {
  pthread_mutex_init(&m_encoder_lock, nullptr);
  pthread_mutex_init(&m_sock_lock, nullptr);
}

WorkerThreads::WorkerThreads(WorkerThreads&& other) noexcept :
  m_worker_ctxs(std::move(other.m_worker_ctxs)),
  m_next_worker(other.m_next_worker),
  m_encoder_lock(other.m_encoder_lock),
  m_encoder(std::move(other.m_encoder)),
  m_sock_lock(other.m_sock_lock),
  m_tcpsock(std::move(other.m_tcpsock)) {

  other.m_encoder_lock = PTHREAD_MUTEX_INITIALIZER;
  other.m_sock_lock = PTHREAD_MUTEX_INITIALIZER;

  for (auto& ctx : m_worker_ctxs) {
    ctx.encoder_lock = &m_encoder_lock;
    ctx.encoder = &m_encoder;

    ctx.sock_lock = &m_sock_lock;
    ctx.tcpsock = &m_tcpsock;
  }
}

WorkerThreads::~WorkerThreads() {
  for (auto& ctx : m_worker_ctxs) {
    pthread_kill(ctx.worker_tid, WORKER_STOP_SIG);
    sem_post(&ctx.launch_task_sem);
  }
  for (auto& ctx : m_worker_ctxs) {
    pthread_join(ctx.worker_tid, nullptr);
  }
  for (auto& ctx : m_worker_ctxs) {
    sem_destroy(&ctx.launch_task_sem);
  }
  pthread_mutex_destroy(&m_encoder_lock);
  pthread_mutex_destroy(&m_sock_lock);
}

void WorkerThreads::launch_workers() {
  for (auto& ctx : m_worker_ctxs) {
    sem_init(&ctx.launch_task_sem, 0, 0);

    ctx.encoder_lock = &m_encoder_lock;
    ctx.encoder = &m_encoder;

    ctx.sock_lock = &m_sock_lock;
    ctx.tcpsock = &m_tcpsock;

    int status = pthread_create(
       &ctx.worker_tid,
       nullptr,
       worker_thread_fn,
       static_cast<void*>(&ctx)
    );
    if (status != 0) {
      std::string err_msg =
        "Failed to spawn worker thread: "
        + std::string(strerror(errno));
      log_(ERROR, err_msg.c_str());
      throw std::runtime_error(err_msg);
    }
  }
}

void WorkerThreads::start_task(
  std::chrono::nanoseconds timestamp,
  std::span<uint8_t> frame
) {
  log_(BENCHMARK, "Starting task");

  m_worker_ctxs[m_next_worker].timestamp = timestamp;
  m_worker_ctxs[m_next_worker].frame = frame;

  sem_post(&m_worker_ctxs[m_next_worker].launch_task_sem);

  m_next_worker = m_next_worker == 0 ? 1 : 0;
}
