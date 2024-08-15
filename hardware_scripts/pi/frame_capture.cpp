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
#include "VFQ.h"

std::atomic<bool> running(true);
int vfq_id = create_vfq(KEY);
Logger& logger = Logger::getLogger("logs.txt");

std::pair<unsigned int, unsigned int> resolution = std::make_pair(1920, 1080);
int buffersCount = 4;
std::pair<std::int64_t, std::int64_t> frameDurationLimits = std::make_pair(16667, 16667);
CameraHandler cam = CameraHandler(resolution, buffersCount, frameDurationLimits);

void sig_handler(int signo, siginfo_t *info, void *context) {
  if (signo == SIGUSR1) {
    cam.queueRequest();
    logger.log(Logger::timestamp(), Logger::Level::INFO, __FILE__, __LINE__, "Capture request queued");
  } else if (signo == SIGINT || signo == SIGTERM) {
    running = false;
  }
}

int main() {
  struct sigaction action;
  int fd;
  pid_t pid = getpid();
  action.sa_sigaction = sig_handler;
  action.sa_flags = SA_SIGINFO;
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGUSR1, &action, NULL) < 0 ||
      sigaction(SIGINT, &action, NULL) < 0 ||
      sigaction(SIGTERM, &action, NULL) < 0) {
    logger.log(Logger::timestamp(), Logger::Level::ERROR, __FILE__, __LINE__, "Failed to set signal handler");
    return -1;
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(3, &cpuset);
  if (sched_setaffinity(0, sizeof(cpuset), &cpuset) < 0) {
    logger.log(Logger::timestamp(), Logger::Level::ERROR, __FILE__, __LINE__, "Failed to set CPU affinity");
    return -1;
  }

  struct sched_param param;
  param.sched_priority = sched_get_priority_max(SCHED_FIFO);
  if (sched_setscheduler(0, SCHED_FIFO, &param) < 0) {
    logger.log(Logger::timestamp(), Logger::Level::ERROR, __FILE__, __LINE__, "Failed to set scheduler");
    return -1;
  }

  fd = open("/proc/gpio_interrupt_pid", O_WRONLY);
  if (fd < 0) {
    logger.log(Logger::timestamp(), Logger::Level::ERROR, __FILE__, __LINE__, "Failed to open /proc/gpio_interrupt_pid");
    return -1;
  }
  if (dprintf(fd, "%d", pid) < 0) {
    logger.log(Logger::timestamp(), Logger::Level::ERROR, __FILE__, __LINE__, "Failed to write to /proc/gpio_interrupt_pid");
    close(fd);
    return -1;
  }
  close(fd);

  while (running) {
    pause();
  }

  destroy_vfq(vfq_id);

  return 0;
}
