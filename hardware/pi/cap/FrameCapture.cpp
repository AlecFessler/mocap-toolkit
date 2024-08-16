#include <cstdlib>
#include <fcntl.h>
#include <memory>
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <signal.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "CameraHandler.h"
#include "Logger.h"
#include "PCtx.h"

constexpr size_t IMAGE_BYTES = 1920 * 1080 * 3;
extern char **environ;

void sig_handler(int signo, siginfo_t *info, void *context) {
  static PCtx& pctx = PCtx::getInstance();
  if (signo == SIGUSR1 && pctx.running) {
    pctx.cam->queueRequest();
    static const char* info = "Capture request queued";
    pctx.logger->log(Logger::Level::INFO, __FILE__, __LINE__, info);
  } else if (signo == SIGINT || signo == SIGTERM) {
    pctx.running = 0;
  }
}

int main() {
  PCtx& pctx = PCtx::getInstance();
  try {
    pctx.logger = std::make_unique<Logger>("cap_logs.txt");
  } catch (const std::exception& e) {
    return -EIO;
  }

  const char* shmName = "/video_frames";
  size_t shmSize = IMAGE_BYTES + sizeof(sem_t);
  int shmFd = shm_open(shmName, O_CREAT | O_RDWR, 0666);
  if (shmFd < 0) {
    const char* err = "Failed to open shared memory";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  if (ftruncate(shmFd, shmSize) < 0) {
    const char* err = "Failed to truncate shared memory";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  void* shmPtr = mmap(nullptr, shmSize, PROT_WRITE, MAP_SHARED, shmFd, 0);
  if (shmPtr == MAP_FAILED) {
    const char* err = "Failed to map shared memory";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }
  close(shmFd);

  sem_t* sem = (sem_t*)((char*)shmPtr);
  if (sem_init(sem, 1, 0) < 0) {
    const char* err = "Failed to initialize semaphore";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  unsigned char* frame = (unsigned char*)((char*)shmPtr + sizeof(sem_t));

  pid_t childPid;
  char* path = (char*)"/home/afessler/Documents/video_capture/framestream";
  char *argv[] = {path, nullptr};
  int spawnStatus = posix_spawn(&childPid, argv[0], nullptr, nullptr, argv, environ);
  if (spawnStatus != 0) {
    const char* err = "Failed to spawn child process";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return spawnStatus;
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(3, &cpuset);
  if (sched_setaffinity(0, sizeof(cpuset), &cpuset) < 0) {
    const char* err = "Failed to set CPU affinity";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  struct sched_param param;
  param.sched_priority = sched_get_priority_max(SCHED_FIFO);
  if (sched_setscheduler(0, SCHED_FIFO, &param) < 0) {
    const char* err = "Failed to set real-time scheduling policy";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  std::pair<unsigned int, unsigned int> resolution = std::make_pair(1920, 1080);
  int buffersCount = 4;
  std::pair<std::int64_t, std::int64_t> frameDurationLimits = std::make_pair(16667, 16667);
  try {
    pctx.cam = std::make_unique<CameraHandler>(resolution, buffersCount, frameDurationLimits);
  } catch (const std::exception& e) {
    const char* err = "Failed to initialize camera";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
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
    const char* err = "Failed to set signal handler";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -EINVAL;
  }

  int fd;
  fd = open("/proc/gpio_interrupt_pid", O_WRONLY);
  if (fd < 0) {
    const char* err = "Failed to open /proc/gpio_interrupt_pid";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }
  if (dprintf(fd, "%d", pid) < 0) {
    const char* err = "Failed to write to /proc/gpio_interrupt_pid";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    close(fd);
    return -errno;
  }
  close(fd);

  pctx.running = 1;
  while (pctx.running) {
    pause();
  }

  munmap(shmPtr, shmSize);
  shm_unlink(shmName);

  return 0;
}
