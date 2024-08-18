#include <fcntl.h>
#include <memory>
#include <sched.h>
#include <semaphore.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ImageConstants.h"
#include "Logger.h"


int main() {
  /*******************************************/
  /* Ensure process is killed if parent dies */
  /*******************************************/

  prctl(PR_SET_PDEATHSIG, SIGINT);

  /*************************/
  /* Initialize the logger */
  /*************************/

  std::unique_ptr<Logger> logger;
  try {
    logger = std::make_unique<Logger>("stream_logs.txt");
  } catch (const std::exception& e) {
    return -EIO;
  }

  /*****************************/
  /* Pin the process to core 2 */
  /*****************************/

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(2, &cpuset);
  if (sched_setaffinity(0, sizeof(cpuset), &cpuset) < 0) {
    const char* err = "Failed to set CPU affinity";
    logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  /***************************************************/
  /* Set scheduling policy to FIFO with max priority */
  /***************************************************/

  struct sched_param param;
  param.sched_priority = sched_get_priority_max(SCHED_FIFO);
  if (sched_setscheduler(0, SCHED_FIFO, &param) < 0) {
    const char* err = "Failed to set real-time scheduling policy";
    logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  /**********************************************************/
  /* Open shared memory to size of 2 semaphores and 1 frame */
  /*                                                        */
  /* shared memory structure:                               */
  /*  - processReady semaphore                              */
  /*  - captureSync semaphore                               */
  /*  - image frame                                         */
  /**********************************************************/

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

  /**********************************/
  /* Map shared memory into process */
  /**********************************/

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

  /************************************/
  /* Set up pointers to shared memory */
  /************************************/

  #define CAST_SHM(type, offset) reinterpret_cast<type>(reinterpret_cast<char*>(shmPtr.get()) + offset)
  sem_t* processReady = CAST_SHM(sem_t*, 0);
  sem_t* captureSync = CAST_SHM(sem_t*, sizeof(sem_t));
  unsigned char* frame = CAST_SHM(unsigned char*, sizeof(sem_t) * 2);

  /*******************************************************/
  /* Signal ready to parent process then wait for frames */
  /*******************************************************/

  sem_post(processReady);
  while (true) {
    sem_wait(captureSync);
    logger->log(Logger::Level::INFO, __FILE__, __LINE__, "Frame received");
    // Process frame
    sem_post(captureSync);
    // prevent tight loop from locking up the semaphore
    usleep(100);
  }

  return 0;
}
