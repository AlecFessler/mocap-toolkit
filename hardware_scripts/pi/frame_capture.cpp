#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <unistd.h>

#include "CameraHandler.h"
#include "Logger.h"
#include "PCtx.h"

void sig_handler(int signo, siginfo_t *info, void *context) {
  static PCtx& pctx = PCtx::getInstance();
  static const char* infostr = "Capture request queued";
  if (signo == SIGUSR1 && pctx.running) {
    pctx.cam->queueRequest();
    pctx.logger->log(Logger::Level::INFO, __FILE__, __LINE__, infostr);
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
    static const char* errstr = "Failed to set signal handler";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, errstr);
    return -EINVAL;
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(3, &cpuset);
  if (sched_setaffinity(0, sizeof(cpuset), &cpuset) < 0) {
    static const char* errstr = "Failed to set CPU affinity, likely due to insufficient permissions";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, errstr);
    return -EPERM;
  }

  struct sched_param param;
  param.sched_priority = sched_get_priority_max(SCHED_FIFO);
  if (sched_setscheduler(0, SCHED_FIFO, &param) < 0) {
    static const char* errstr = "Failed to set scheduler policy, likely due to insufficient permissions";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, errstr);
    return -EPERM;
  }

  int fd;
  fd = open("/proc/gpio_interrupt_pid", O_WRONLY);
  if (fd < 0) {
    static const char* errstr = "Failed to open /proc/gpio_interrupt_pid";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, errstr);
    return -errno;
  }
  if (dprintf(fd, "%d", pid) < 0) {
    static const char* errstr = "Failed to write to /proc/gpio_interrupt_pid";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, errstr);
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
    static const char* errstr = "Failed to initialize camera";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, errstr);
    return -EIO;
  }

  pctx.running = 1;
  while (pctx.running) {
    pause();
  }

  return 0;
}
