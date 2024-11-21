// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <atomic>
#include <arpa/inet.h>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <sched.h>
#include <semaphore.h>
#include <signal.h>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>
#include "camera_handler.h"
#include "logger.h"
#include "videnc.h"

extern char** environ;
volatile static sig_atomic_t running = 0;
static timer_t timerid;
static int sockfd = -1;
std::unique_ptr<logger_t> logger;
std::unique_ptr<camera_handler_t> cam;

void sig_handler(int signo, siginfo_t* info, void* context) {
  (void)info;
  (void)context;
  if (signo == SIGUSR1 && running) {
    cam->queue_request();
    logger->log(logger_t::level_t::INFO, __FILE__, __LINE__, "Capture request queued");
  } else if (signo == SIGINT || signo == SIGTERM) {
    running = 0;
    // handle the rest of the frames before exiting
  }
}

int main() {
  try {
    config_parser config = config_parser("config.txt");
    std::string server_ip = config.get_string("SERVER_IP");
    std::string port = config.get_string("PORT");
    int frame_buffers = config.get_int("FRAME_BUFFERS");
    int recording_cpu = config.get_int("RECORDING_CPU");

    logger = std::make_unique<logger_t>("logs.txt");

    lock_free_queue_t frame_queue(frame_buffers);

    sem_t queue_counter;
    if (sem_init(&queue_counter, 0, 0) < 0) {
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to initialize semaphore");
      return -errno;
    }

    cam = std::make_unique<camera_handler_t>(config, *logger, frame_queue, queue_counter);
    videnc encoder(config);

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(recording_cpu, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) < 0) {
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to set CPU affinity");
      return -errno;
    }

    struct sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    if (sched_setscheduler(0, SCHED_FIFO, &param) < 0) {
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to set real-time scheduling policy");
      return -errno;
    }

    sockfd = socket(AF_INET, SOCK_STREAM, 0)
    if (sockfd < 0) {
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to create socket");
      return -errno;
    }

    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(std::stoi(port));
    inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr);

    if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to connect to server");
      return -errno;
    }

    struct sigevent sev;
    sev.sigev_notify = SIGEV_SIGNAL;
    sev.sigev_signo = SIGUSR2;
    sev.sigev_value.sival_ptr = &timerid;

    if (timer_create(CLOCK_MONOTONIC, &sev, &timerid) == -1) {
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to create socket disconnect timer");
      return -errno;
    }

    int fd;
    fd = open("/proc/gpio_interrupt_pid", O_WRONLY);
    if (fd < 0) {
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to open /proc/gpio_interrupt_pid");
      return -errno;
    }
    pid_t pid = getpid();
    if (dprintf(fd, "%d", pid) < 0) {
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to write to /proc/gpio_interrupt_pid");
      close(fd);
      return -errno;
    }
    close(fd);

    struct sigaction action;
    action.sa_sigaction = sig_handler;
    action.sa_flags = SA_SIGINFO;
    sigemptyset(&action.sa_mask);
    if (
      sigaction(SIGUSR1, &action, NULL) < 0 ||
      sigaction(SIGINT, &action, NULL) < 0 ||
      sigaction(SIGTERM, &action, NULL) < 0
    ) {
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to set signal handler");
      return -errno;
    }

    running = 1;
    while (running) {
      sem_wait(&queue_counter);

      void* frame = frame_queue.dequeue();
      if (!frame) continue;

      encoder.encode_frame((uint8_t*)frame);

      size_t enc_bytes = encoder.pkt->size;
      if (enc_bytes == 0) {
        logger->log(logger_t::level_t::DEBUG, __FILE__, __LINE__, "No frame data");
        continue;
      }

      size_t total_bytes_written = 0;
      while (total_bytes_written < enc_bytes) {
        ssize_t result = write(
          *socket_ptr,
          encoder.pkt->data + total_bytes_written,
          enc_bytes - total_bytes_written
        );

        if (result < 0) {
          if (errno == EINTR) continue;
          logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Error transmitting frame");
          running = 0;
          break;
        }

        total_bytes_written += result;
      }

      logger->log(logger_t::level_t::INFO, __FILE__, __LINE__, "Transmitted frame");
    }

  } catch (const std::exception& e) {
    if (logger)
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, e.what());
    else
      std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
