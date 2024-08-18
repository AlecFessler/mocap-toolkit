#include <fcntl.h>
#include <memory>
#include <sched.h>
#include <semaphore.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "logger.h"
#include "shared_defs.h"
#include "tcp_thread.h"


int main() {
  /*******************************************/
  /* Ensure process is killed if parent dies */
  /*******************************************/

  prctl(PR_SET_PDEATHSIG, SIGINT);

  /*************************/
  /* Initialize the logger */
  /*************************/

  std::unique_ptr<logger_t> logger;
  try {
    logger = std::make_unique<logger_t>("stream_logs.txt");
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
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  /***************************************************/
  /* Set scheduling policy to FIFO with max priority */
  /***************************************************/

  struct sched_param param;
  param.sched_priority = sched_get_priority_max(SCHED_FIFO);
  if (sched_setscheduler(0, SCHED_FIFO, &param) < 0) {
    const char* err = "Failed to set real-time scheduling policy";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
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
  /*  - process_ready semaphore                             */
  /*  - img_write_sem semaphore                             */
  /*  - img_read_sem semaphore                              */
  /*  - image frame                                         */
  /**********************************************************/

  const char* shared_mem_name = "/video_frames";
  size_t shared_mem_size = IMAGE_BYTES + sizeof(sem_t) * 3;
  int shared_mem_fd = shm_open(shared_mem_name, O_RDWR, 0666);
  if (shared_mem_fd < 0) {
    const char* err = "Failed to open shared memory";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  if (ftruncate(shared_mem_fd, shared_mem_size) < 0) {
    const char* err = "Failed to truncate shared memory";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    close(shared_mem_fd);
    return -errno;
  }

  /**********************************/
  /* Map shared memory into process */
  /**********************************/

  auto mmap_cleanup = [shared_mem_size](void* ptr) {
    if (ptr && ptr != MAP_FAILED) {
      munmap(ptr, shared_mem_size);
    }
  };
  std::unique_ptr<void, decltype(mmap_cleanup)> shared_mem_ptr(mmap(nullptr, shared_mem_size, PROT_READ | PROT_WRITE, MAP_SHARED, shared_mem_fd, 0), mmap_cleanup);
  if (shared_mem_ptr.get() == MAP_FAILED) {
    const char* err = "Failed to map shared memory";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    close(shared_mem_fd);
    return -errno;
  }
  close(shared_mem_fd);

  /************************************/
  /* Set up pointers to shared memory */
  /************************************/

  sem_t* process_ready = PTR_MATH_CAST(sem_t, shared_mem_ptr.get(), 0);
  sem_t* img_write_sem = PTR_MATH_CAST(sem_t, shared_mem_ptr.get(), sizeof(sem_t));
  sem_t* img_read_sem = PTR_MATH_CAST(sem_t, shared_mem_ptr.get(), 2 * sizeof(sem_t));
  unsigned char* img_data = PTR_MATH_CAST(unsigned char, shared_mem_ptr.get(), 3 * sizeof(sem_t));

  /*******************************************************/
  /* Signal ready to parent process then wait for frames */
  /*******************************************************/

  sem_post(process_ready);
  while (true) {
    sem_wait(img_read_sem);
    logger->log(logger_t::level_t::INFO, __FILE__, __LINE__, "Frame received");
    pthread_mutex_lock(&child_thread_data.mutex);
    // push image to shared queue
    sem_post(img_write_sem);
    pthread_mutex_unlock(&child_thread_data.mutex);
  }

  return 0;
}
