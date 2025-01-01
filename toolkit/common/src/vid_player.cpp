#include <csignal>
#include <cstring>
#include <cerrno>
#include <opencv2/opencv.hpp>
#include <semaphore.h>
#include <spsc_queue.hpp>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "logging.h"
#include "sem_init.hpp"
#include "vid_player.hpp"

constexpr uint64_t NS_PER_S = 1'000'000'000;

static volatile sig_atomic_t running = 1;
static std::unique_ptr<sem_t, sem_deleter> loop_ctl_sem;

static void display_handler(int signum);
static void shutdown_handler(int signum);

void* display_thread_fn(void* ptr) {
  int ret = 0;
  char logstr[128];

  struct display_thread_ctx* ctx = (struct display_thread_ctx*)ptr;

  struct sigaction shutdown_sa;
  shutdown_sa.sa_handler = shutdown_handler;
  shutdown_sa.sa_flags = 0;
  sigemptyset(&shutdown_sa.sa_mask);
  sigaction(SIGUSR2, &shutdown_sa, NULL);

  struct sigaction display_sa;
  display_sa.sa_handler = display_handler;
  display_sa.sa_flags = 0;
  sigemptyset(&display_sa.sa_mask);
  sigaction(SIGUSR1, &display_sa, NULL);

  loop_ctl_sem = init_semaphore();

  timer_t timerid;
  struct sigevent sev;
  sev.sigev_notify = SIGEV_SIGNAL;
  sev.sigev_signo = SIGUSR1;
  sev.sigev_value.sival_ptr = &timerid;
  ret = timer_create(CLOCK_MONOTONIC, &sev, &timerid);
  if (ret == -1) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Failed to create timer: %s",
      strerror(errno)
    );
    log_write(ERROR, logstr);
  }

  struct itimerspec its;
  its.it_value.tv_sec = 0;
  its.it_value.tv_nsec = NS_PER_S / ctx->stream_conf->fps;
  its.it_interval.tv_sec = 0;
  its.it_interval.tv_nsec = 0;

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(ctx->core, &cpuset);
  sched_setaffinity(
    gettid(),
    sizeof(cpu_set_t),
    &cpuset
  );

  std::string window_name;

  while (running) {
    cv::Mat* bgr_frames = static_cast<cv::Mat*>(
      spsc_dequeue(ctx->filled_frameset_q)
    );
    if (bgr_frames == nullptr) {
      usleep(100); // 0.1ms
      continue;
    }

    for (uint32_t i = 0; i < ctx->num_frames; i++) {
      window_name = "stream" + std::to_string(i);
      cv::imshow(window_name, bgr_frames[i]);
      cv::waitKey(1);
    }

    spsc_enqueue(
      ctx->empty_frameset_q,
      bgr_frames
    );

    timer_settime(
      timerid,
      /*relative timer*/ 0,
      &its,
      /*no old val*/ nullptr
    );

    sem_wait(loop_ctl_sem.get());
  }

  cv::destroyAllWindows();
  return nullptr;
}

static void display_handler(int signum) {
  (void)signum;
  sem_post(loop_ctl_sem.get());
}

static void shutdown_handler(int signum) {
  (void)signum;
  running = 0;
  sem_post(loop_ctl_sem.get());
}
