#include <cstring>
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

#include "logger.h"
#include "shared_defs.h"
#include "tcp_thread.h"


int main() {
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

  /******************************************/
  /* Open memory shared with parent process */
  /******************************************/

  const char* shared_mem_name = "/video_frames";
  int shared_mem_fd = shm_open(shared_mem_name, O_RDWR, 0666);
  if (shared_mem_fd < 0) {
    const char* err = "Failed to open shared memory";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  if (ftruncate(shared_mem_fd, sizeof(shared_mem_layout)) < 0) {
    const char* err = "Failed to truncate shared memory";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    close(shared_mem_fd);
    return -errno;
  }

  /**********************************/
  /* Map shared memory into process */
  /**********************************/

  auto mmap_cleanup = [](void* ptr) {
    if (ptr && ptr != MAP_FAILED) {
      munmap(ptr, sizeof(shared_mem_layout));
    }
  };
  std::unique_ptr<void, decltype(mmap_cleanup)> shared_mem_ptr(mmap(nullptr, sizeof(shared_mem_layout), PROT_READ | PROT_WRITE, MAP_SHARED, shared_mem_fd, 0), mmap_cleanup);
  if (shared_mem_ptr.get() == MAP_FAILED) {
    const char* err = "Failed to map shared memory";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    close(shared_mem_fd);
    return -errno;
  }
  shared_mem_layout& shared_mem = *static_cast<shared_mem_layout*>(shared_mem_ptr.get());
  close(shared_mem_fd);

  /********************************/
  /* Get handles to shared memory */
  /********************************/

  volatile sig_atomic_t& running = shared_mem.running;
  sem_t& parent_process_ready = shared_mem.parent_process_ready_sem;
  sem_t& this_process_ready = shared_mem.child_process_ready_sem;
  sem_t& img_read_sem = shared_mem.img_read_sem;
  sem_t& img_write_sem = shared_mem.img_write_sem;
  unsigned char* img_data = shared_mem.img_data;

  /******************************************************/
  /* Initialize child thread and shared data structures */
  /******************************************************/

  shared_data child_thread_data(running);
  child_thread_data.running = running;
  lock_free_queue_t& queue = child_thread_data.queue;

  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);
  pthread_mutex_init(&child_thread_data.mutex, &attr);
  pthread_mutex_t& mutex = child_thread_data.mutex;

  pthread_cond_init(&child_thread_data.cond, nullptr);

  pthread_t child_thread;
  pthread_create(&child_thread, nullptr, tcp_thread, &child_thread_data);

  /*******************************************************/
  /* Signal ready to parent process then wait for frames */
  /*******************************************************/

  sem_post(&this_process_ready);
  sem_wait(&parent_process_ready);
  while (running) {
    sem_wait(&img_read_sem);
    logger->log(logger_t::level_t::INFO, __FILE__, __LINE__, "Frame received");

    auto img_buffer = std::make_unique<unsigned char[]>(IMAGE_BYTES);
    if (!img_buffer) {
      const char* err = "Failed to allocate memory for image buffer";
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    }
    memcpy(img_buffer.get(), img_data, IMAGE_BYTES);

    queue.push(img_buffer.release());
    pthread_cond_signal(&child_thread_data.cond);

    sem_post(&img_write_sem);
  }

  return 0;
}
