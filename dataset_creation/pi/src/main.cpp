// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <cstdint>
#include <exception>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <sched.h>
#include <signal.h>
#include "camera_handler.h"
#include "config_parser.h"
#include "logger.h"

extern char** environ;

std::unique_ptr<logger_t> logger;
std::unique_ptr<camera_handler_t> cam;
static bool running = false;

void sig_handler(int signo, siginfo_t* info, void* context) {
  (void)info;
  (void)context;
  if (signo == SIGUSR1 && running) {
    cam->queue_request();
    const char* info = "Capture request queued";
    logger->log(logger_t::level_t::INFO, __FILE__, __LINE__, info);
  } else if (signo == SIGINT || signo == SIGTERM) {
    running = false;
  }
}

int main() {
  try {
    config_parser config = config_parser("config.txt");
    logger = std::make_unique<logger_t>("logs.txt");
    cam = std::make_unique<camera_handler_t>(config, *logger);

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(config.get_int("RECORDING_CPU"), &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) < 0) {
      const char* err = "Failed to set CPU affinity";
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
      return -errno;
    }
  } catch (const std::exception& e) {
    if (logger)
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, e.what());
    else
      std::cerr << e.what() << std::endl;
    return 1;
  }


  struct sched_param param;
  param.sched_priority = sched_get_priority_max(SCHED_FIFO);
  if (sched_setscheduler(0, SCHED_FIFO, &param) < 0) {
    const char* err = "Failed to set real-time scheduling policy";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  struct sigaction action;
  action.sa_sigaction = sig_handler;
  action.sa_flags = SA_SIGINFO;
  sigemptyset(&action.sa_mask);
  if (
    sigaction(SIGUSR1, &action, NULL) < 0 ||
    sigaction(SIGINT, &action, NULL) < 0 ||
    sigaction(SIGTERM, &action, NULL) < 0
  ) {
    const char* err = "Failed to set signal handler";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  int fd;
  fd = open("/proc/gpio_interrupt_pid", O_WRONLY);
  if (fd < 0) {
    const char* err = "Failed to open /proc/gpio_interrupt_pid";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }
  pid_t pid = getpid();
  if (dprintf(fd, "%d", pid) < 0) {
    const char* err = "Failed to write to /proc/gpio_interrupt_pid";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    close(fd);
    return -errno;
  }
  close(fd);

  running = true;
  while (running) {
    pause();
  }

  return 0;
}
