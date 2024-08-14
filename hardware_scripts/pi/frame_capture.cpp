#include <signal.h>
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <fcntl.h>
#include <sched.h>
#include <pthread.h>
#include <atomic>
#include <memory>

#include "CameraHandler.h"
#include "VFQ.h"

std::atomic<bool> running(true);
int vfq_id = create_vfq(KEY);
std::pair<unsigned int, unsigned int> resolution = std::make_pair(1920, 1080);
int buffersCount = 4;
std::pair<std::int64_t, std::int64_t> frameDurationLimits = std::make_pair(16667, 16667);
CameraHandler cam = CameraHandler(resolution, buffersCount, frameDurationLimits);

void sig_handler(int signo, siginfo_t *info, void *context) {
  if (signo == SIGUSR1) {
    cam.QueueRequest();
  } else if (signo == SIGINT || signo == SIGTERM) {
    running = false;
    std::cout << "Received signal " << signo << ", exiting..." << std::endl;
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
    std::cerr << "Failed to register signal handlers" << std::endl;
    return -1;
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(3, &cpuset);
  if (sched_setaffinity(0, sizeof(cpuset), &cpuset) < 0) {
    std::cerr << "Failed to set CPU affinity" << std::endl;
    return -1;
  }

  struct sched_param param;
  param.sched_priority = sched_get_priority_max(SCHED_FIFO);
  if (sched_setscheduler(0, SCHED_FIFO, &param) < 0) {
    std::cerr << "Failed to set real-time scheduling" << std::endl;
    return -1;
  }

  fd = open("/proc/gpio_interrupt_pid", O_WRONLY);
  if (fd < 0) {
    std::cerr << "Failed to open /proc/gpio_interrupt_pid" << std::endl;
    return -1;
  }
  if (dprintf(fd, "%d", pid) < 0) {
    std::cerr << "Failed to write to /proc/gpio_interrupt_pid" << std::endl;
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
