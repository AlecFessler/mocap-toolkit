#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <string>
#include <unistd.h>

#include "CameraHandler.h"
#include "Logger.h"
#include "PCtx.h"

void sig_handler(int signo, siginfo_t *info, void *context) {
  static PCtx& pctx = PCtx::getInstance();
  if (signo == SIGUSR1 && pctx.running) {
    pctx.cam->queueRequest();
    pctx.logger->queue(Logger::timestamp(), Logger::Level::INFO, __FILE__, __LINE__, "Capture request queued");
  } else if (signo == SIGINT || signo == SIGTERM) {
    pctx.running = 0;
  }
}

int main() {
  PCtx& pctx = PCtx::getInstance();

  try {
    pctx.logger = std::make_unique<Logger>("logs.txt");
  } catch (const std::exception& e) {
    return -EIO;
  }

  struct sigaction action;
  pid_t pid = getpid();
  action.sa_sigaction = sig_handler;
  action.sa_flags = SA_SIGINFO;
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGUSR1, &action, NULL) < 0 ||
      sigaction(SIGINT, &action, NULL) < 0 ||
      sigaction(SIGTERM, &action, NULL) < 0) {
    pctx.logger->log(Logger::timestamp(), Logger::Level::ERROR, __FILE__, __LINE__, "Failed to set signal handler");
    return -EINVAL;
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(3, &cpuset);
  if (sched_setaffinity(0, sizeof(cpuset), &cpuset) < 0) {
    pctx.logger->log(Logger::timestamp(), Logger::Level::ERROR, __FILE__, __LINE__, "Failed to set CPU affinity, likely due to insufficient permissions");
    return -EPERM;
  }

  struct sched_param param;
  param.sched_priority = sched_get_priority_max(SCHED_FIFO);
  if (sched_setscheduler(0, SCHED_FIFO, &param) < 0) {
    pctx.logger->log(Logger::timestamp(), Logger::Level::ERROR, __FILE__, __LINE__, "Failed to set scheduler policy, likely due to insufficient permissions");
    return -EPERM;
  }

  int fd;
  fd = open("/proc/gpio_interrupt_pid", O_WRONLY);
  if (fd < 0) {
    pctx.logger->log(Logger::timestamp(), Logger::Level::ERROR, __FILE__, __LINE__, "Failed to open /proc/gpio_interrupt_pid");
    return -errno;
  }
  if (dprintf(fd, "%d", pid) < 0) {
    pctx.logger->log(Logger::timestamp(), Logger::Level::ERROR, __FILE__, __LINE__, "Failed to write to /proc/gpio_interrupt_pid");
    close(fd);
    return -errno;
  }
  close(fd);

  std::pair<unsigned int, unsigned int> resolution = std::make_pair(1920, 1080);
  int buffersCount = 4;
  std::pair<std::int64_t, std::int64_t> frameDurationLimits = std::make_pair(16667, 16667);
  try {
    pctx.cam = std::make_unique<CameraHandler>(resolution, buffersCount, frameDurationLimits);
  } catch (const std::exception& e) {
    pctx.logger->log(Logger::timestamp(), Logger::Level::ERROR, __FILE__, __LINE__, "Failed to initialize camera");
    return -EIO;
  }

  pctx.running = 1;
  while (pctx.running) {
    pause();
  }

  return 0;
}
