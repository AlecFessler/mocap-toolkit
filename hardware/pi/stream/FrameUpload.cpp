#include <fcntl.h>
#include <memory>
#include <sched.h>
#include <semaphore.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>

#include "Logger.h"

constexpr size_t IMAGE_BYTES = 1920 * 1080 * 3;

int main() {
  prctl(PR_SET_PDEATHSIG, SIGINT);

  std::unique_ptr<Logger> logger;
  try {
    logger = std::make_unique<Logger>("stream_logs.txt");
  } catch (const std::exception& e) {
    return -EIO;
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(2, &cpuset);
  if (sched_setaffinity(0, sizeof(cpuset), &cpuset) < 0) {
    const char* err = "Failed to set CPU affinity";
    logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  struct sched_param param;
  param.sched_priority = sched_get_priority_max(SCHED_FIFO);
  if (sched_setscheduler(0, SCHED_FIFO, &param) < 0) {
    const char* err = "Failed to set real-time scheduling policy";
    logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  const char* shmName = "/video_frames";
  size_t shmSize = IMAGE_BYTES + sizeof(sem_t) * 2;
  int shmFd = shm_open(shmName, O_RDWR, 0666);
  if (shmFd < 0) {
    const char* err = "Failed to open shared memory";
    logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  if (ftruncate(shmFd, shmSize) < 0) {
    const char* err = "Failed to truncate shared memory";
    logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    close(shmFd);
    return -errno;
  }

  auto munmapDeleter = [shmSize](void* ptr) {
    if (ptr && ptr != MAP_FAILED) {
      munmap(ptr, shmSize);
    }
  };
  std::unique_ptr<void, decltype(munmapDeleter)> shmPtr(mmap(nullptr, shmSize, PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0), munmapDeleter);
  if (shmPtr.get() == MAP_FAILED) {
    const char* err = "Failed to map shared memory";
    logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    close(shmFd);
    return -errno;
  }
  close(shmFd);

#define CAST_SHM(type, offset) reinterpret_cast<type>(reinterpret_cast<char*>(shmPtr.get()) + offset)
  sem_t* processReady = CAST_SHM(sem_t*, 0);
  sem_t* captureSync = CAST_SHM(sem_t*, sizeof(sem_t));
  unsigned char* frame = CAST_SHM(unsigned char*, sizeof(sem_t) * 2);

  sem_post(processReady);
  while (true) {
    sem_wait(captureSync);
    logger->log(Logger::Level::INFO, __FILE__, __LINE__, "Frame received");
    sem_post(captureSync);
    usleep(100);
  }

  return 0;
}
