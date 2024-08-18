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

#include "camera_handler.h"
#include "shared_defs.h"
#include "logger.h"
#include "p_ctx.h"

extern char** environ;

void sig_handler(int signo, siginfo_t* info, void* context) {
  static p_ctx_t& p_ctx = p_ctx_t::get_instance();
  if (signo == SIGUSR1 && p_ctx.running) {
    p_ctx.cam->queue_request();
    static const char* info = "Capture request queued";
    p_ctx.logger->log(logger_t::level_t::INFO, __FILE__, __LINE__, info);
  } else if (signo == SIGINT || signo == SIGTERM) {
    p_ctx.running = 0;
  }
}

int main() {

  /*************************/
  /* Initialize the logger */
  /*************************/

  p_ctx_t& p_ctx = p_ctx_t::get_instance();
  try {
    p_ctx.logger = std::make_unique<logger_t>("cap_logs.txt");
  } catch (const std::exception& e) {
    return -EIO;
  }

  /*************************/
  /* Initialize the camera */
  /*************************/

  std::pair<unsigned int, unsigned int> resolution = std::make_pair(IMAGE_WIDTH, IMAGE_HEIGHT);
  int buffers_count = 4;
  std::pair<std::int64_t, std::int64_t> frame_duration_limits = std::make_pair(16667, 16667);
  try {
    p_ctx.cam = std::make_unique<camera_handler_t>(resolution, buffers_count, frame_duration_limits);
  } catch (const std::exception& e) {
    const char* err = "Failed to initialize camera";
    p_ctx.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    return -EIO;
  }

  /************************************************************************/
  /* Initialize shared memory to the size of the image and two semaphores */
  /************************************************************************/

  const char* shared_mem_name = "/video_frames";
  size_t shared_mem_size = IMAGE_BYTES + sizeof(sem_t) * 3;
  auto shared_mem_cleanup = [shared_mem_name](int* fd) {
    if (fd && *fd >= 0) {
      close(*fd);
      shm_unlink(shared_mem_name);
    }
    delete fd;
  };
  int shared_mem_fd = shm_open(shared_mem_name, O_CREAT | O_RDWR, 0666);
  if (shared_mem_fd < 0) {
    const char* err = "Failed to open shared memory";
    p_ctx.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }
  std::unique_ptr<int, decltype(shared_mem_cleanup)> shared_mem_fd_ptr(new int(shared_mem_fd), shared_mem_cleanup);

  if (ftruncate(*shared_mem_fd_ptr.get(), shared_mem_size) < 0) {
    const char* err = "Failed to truncate shared memory";
    p_ctx.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  /****************************************/
  /* Map the shared memory to the process */
  /****************************************/

  auto mmap_cleanup = [shared_mem_size](void* ptr) {
    if (ptr && ptr != MAP_FAILED)
      munmap(ptr, shared_mem_size);
  };
  std::unique_ptr<void, decltype(mmap_cleanup)> shared_mem_ptr(mmap(nullptr, shared_mem_size, PROT_READ | PROT_WRITE, MAP_SHARED, *shared_mem_fd_ptr.get(), 0), mmap_cleanup);
  if (shared_mem_ptr.get() == MAP_FAILED) {
    const char* err = "Failed to map shared memory";
    p_ctx.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }
  p_ctx.shared_mem = shared_mem_ptr.get();

  /*****************************/
  /* Initialize the semaphores */
  /*****************************/

  sem_t* child_process_ready = PTR_MATH_CAST(sem_t, shared_mem_ptr.get(), 0);
  if (sem_init(child_process_ready, 1, 0) < 0) {
    const char* err = "Failed to initialize child process ready semaphore";
    p_ctx.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  sem_t* img_write_sem = PTR_MATH_CAST(sem_t, shared_mem_ptr.get(), sizeof(sem_t));
  if (sem_init(img_write_sem, 1, 1) < 0) {
    const char* err = "Failed to initialize image write semaphore";
    p_ctx.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  sem_t* img_read_sem = PTR_MATH_CAST(sem_t, shared_mem_ptr.get(), sizeof(sem_t) * 2);
  if (sem_init(img_read_sem, 1, 0) < 0) {
    const char* err = "Failed to initialize image read semaphore";
    p_ctx.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  /***************************/
  /* Spawn the child process */
  /***************************/

  pid_t child_pid;
  char* path = (char*)"/home/afessler/Documents/video_capture/framestream";
  char* argv[] = {path, nullptr};
  int spawn_status = posix_spawn(&child_pid, argv[0], nullptr, nullptr, argv, environ);
  if (spawn_status != 0) {
    const char* err = "Failed to spawn child process";
    p_ctx.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    return spawn_status;
  }

  /*****************************/
  /* Pin the process to core 3 */
  /*****************************/

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(3, &cpuset);
  if (sched_setaffinity(0, sizeof(cpuset), &cpuset) < 0) {
    const char* err = "Failed to set CPU affinity";
    p_ctx.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  /*******************************************************/
  /* Set the scheduling policy to FIFO with max priority */
  /*******************************************************/

  struct sched_param param;
  param.sched_priority = sched_get_priority_max(SCHED_FIFO);
  if (sched_setscheduler(0, SCHED_FIFO, &param) < 0) {
    const char* err = "Failed to set real-time scheduling policy";
    p_ctx.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  /**************************/
  /* Set the signal handler */
  /**************************/

  struct sigaction action;
  pid_t pid = getpid();
  action.sa_sigaction = sig_handler;
  action.sa_flags = SA_SIGINFO;
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGUSR1, &action, NULL) < 0 ||
      sigaction(SIGINT, &action, NULL) < 0 ||
      sigaction(SIGTERM, &action, NULL) < 0) {
    const char* err = "Failed to set signal handler";
    p_ctx.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  /***********************************************/
  /* Register the process with the kernel module */
  /***********************************************/

  int fd;
  fd = open("/proc/gpio_interrupt_pid", O_WRONLY);
  if (fd < 0) {
    const char* err = "Failed to open /proc/gpio_interrupt_pid";
    p_ctx.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }
  if (dprintf(fd, "%d", pid) < 0) {
    const char* err = "Failed to write to /proc/gpio_interrupt_pid";
    p_ctx.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    close(fd);
    return -errno;
  }
  close(fd);

  /***************************************************************/
  /* Wait for the child process, then begin and wait for signals */
  /***************************************************************/

  sem_wait(child_process_ready);
  p_ctx.running = 1;
  while (p_ctx.running) {
    pause();
  }

  return 0;
}
