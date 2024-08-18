#include <fcntl.h>
#include <memory>
#include <sched.h>
#include <semaphore.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "Logger.h"
#include "SharedDefs.h"
#include "tcp_thread.h"


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

  /******************************************************/
  /* Initialize child thread and shared queue and mutex */
  /******************************************************/

  shared_data child_thread_data;
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);
  pthread_mutex_init(&child_thread_data.mutex, &attr);

  pthread_t child_thread;
  pthread_create(&child_thread, nullptr, tcp_thread, &child_thread_data);

  /**********************************************************/
  /* Open memory shared with parent process                 */
  /*                                                        */
  /* shared memory structure:                               */
  /*  - processReady semaphore                              */
  /*  - imgWriteSem semaphore                               */
  /*  - imgReadSem semaphore                                */
  /*  - image frame                                         */
  /**********************************************************/

  const char* shmName = "/video_frames";
  size_t shmSize = IMAGE_BYTES + sizeof(sem_t) * 3;
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

  sem_t* processReady = PTR_MATH_CAST(sem_t, shmPtr.get(), 0);
  sem_t* imgWriteSem = PTR_MATH_CAST(sem_t, shmPtr.get(), sizeof(sem_t));
  sem_t* imgReadSem = PTR_MATH_CAST(sem_t, shmPtr.get(), 2 * sizeof(sem_t));
  unsigned char* frame = PTR_MATH_CAST(unsigned char, shmPtr.get(), 3 * sizeof(sem_t));

  /*******************************************************/
  /* Signal ready to parent process then wait for frames */
  /*******************************************************/

  sem_post(processReady);
  while (true) {
    sem_wait(imgReadSem);
    logger->log(Logger::Level::INFO, __FILE__, __LINE__, "Frame received");
    pthread_mutex_lock(&child_thread_data.mutex);
    // push image to shared queue
    sem_post(imgWriteSem);
    pthread_mutex_unlock(&child_thread_data.mutex);
  }

  return 0;
}
