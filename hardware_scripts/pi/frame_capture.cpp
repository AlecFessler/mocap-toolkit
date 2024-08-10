#include <signal.h>
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <fcntl.h>
#include <sstream>
#include <sched.h>
#include <pthread.h>

//#include "VFQ.h"

void sig_handler(int signo, siginfo_t *info, void *context) {
  if (signo == SIGUSR1) {
    std::cout << "Received SIGUSR1 signal" << std::endl;
  }
}

int main() {
  // Setup signal handler
  struct sigaction action;
  int fd;
  pid_t pid = getpid();
  action.sa_sigaction = sig_handler;
  action.sa_flags = SA_SIGINFO;
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGUSR1, &action, NULL) < 0) {
    std::cerr << "Failed to register signal handler" << std::endl;
    return -1;
  }

  // Pin the process to core 3
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(3, &cpuset);
  if (sched_setaffinity(0, sizeof(cpuset), &cpuset) < 0) {
    std::cerr << "Failed to set CPU affinity" << std::endl;
    return -1;
  }

  // Set up real-time scheduling with FIFO and max-priority
  struct sched_param param;
  param.sched_priority = sched_get_priority_max(SCHED_FIFO);
  if (sched_setscheduler(0, SCHED_FIFO, &param) < 0) {
    std::cerr << "Failed to set real-time scheduling" << std::endl;
    return -1;
  }

  // Register the process ID with the kernel module
  fd = open("/proc/gpio_interrupt_pid", O_WRONLY);
  if (fd < 0) {
    std::cerr << "Failed to open /proc/gpio_interrupt_pid" << std::endl;
    return -1;
  }
  std::ostringstream oss;
  oss << pid;
  std::string pid_str = oss.str();
  ssize_t bytes_written = write(fd, pid_str.c_str(), pid_str.length());
  if (bytes_written < 0) {
      std::cerr << "Failed to write PID to /proc/gpio_interrupt_pid" << std::endl;
      close(fd);
      return -1;
  } else if (bytes_written != static_cast<ssize_t>(pid_str.length())) {
      std::cerr << "Incomplete write of PID to /proc/gpio_interrupt_pid" << std::endl;
      close(fd);
      return -1;
  }

  while (1) {
    pause(); // Wait indefinitely for signal
  }

  return 0;
}
