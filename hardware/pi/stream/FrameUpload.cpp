#include <fcntl.h>
#include <iostream>
#include <memory>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "Logger.h"

constexpr size_t IMAGE_BYTES = 1920 * 1080 * 3;

int main() {
  std::unique_ptr<Logger> logger;
  try {
    logger = std::make_unique<Logger>("stream_logs.txt");
  } catch (const std::exception& e) {
    return -EIO;
  }

  const char* shmName = "/video_frames";
  size_t shmSize = IMAGE_BYTES + sizeof(sem_t);
  int shmFd = shm_open(shmName, O_RDWR, 0666);
  if (shmFd < 0) {
    const char* err = "Failed to open shared memory";
    logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  if (ftruncate(shmFd, shmSize) < 0) {
    const char* err = "Failed to truncate shared memory";
    logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  void* shmPtr = mmap(nullptr, shmSize, PROT_READ, MAP_SHARED, shmFd, 0);
  if (shmPtr == MAP_FAILED) {
    const char* err = "Failed to map shared memory";
    logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }
  close(shmFd);

  sem_t* sem = (sem_t*)((char*)shmPtr);
  unsigned char* frame = (unsigned char*)((char*)shmPtr + sizeof(sem_t));

  while (true) {
    sem_wait(sem);
    // Process frame
    sem_post(sem);
  }

  munmap(shmPtr, shmSize);

  return 0;
}
