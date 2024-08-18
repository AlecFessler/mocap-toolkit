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
#include "ImageConstants.h"
#include "Logger.h"
#include "PCtx.h"

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

  /*************************/
  /* Initialize the logger */
  /*************************/

  PCtx& pctx = PCtx::getInstance();
  try {
    pctx.logger = std::make_unique<Logger>("cap_logs.txt");
  } catch (const std::exception& e) {
    return -EIO;
  }

  /*************************/
  /* Initialize the camera */
  /*************************/

  std::pair<unsigned int, unsigned int> resolution = std::make_pair(IMAGE_WIDTH, IMAGE_HEIGHT);
  int buffersCount = 4;
  std::pair<std::int64_t, std::int64_t> frameDurationLimits = std::make_pair(16667, 16667);
  try {
    pctx.cam = std::make_unique<CameraHandler>(resolution, buffersCount, frameDurationLimits);
  } catch (const std::exception& e) {
    const char* err = "Failed to initialize camera";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -EIO;
  }

  /************************************************************************/
  /* Initialize shared memory to the size of the image and two semaphores */
  /************************************************************************/

  const char* shmName = "/video_frames";
  size_t shmSize = IMAGE_BYTES + sizeof(sem_t) * 2;
  auto shmDeleter = [shmName](int* fd) {
    if (fd && *fd >= 0) {
      close(*fd);
      shm_unlink(shmName);
    }
    delete fd;
  };
  int shm_fd = shm_open(shmName, O_CREAT | O_RDWR, 0666);
  if (shm_fd < 0) {
    const char* err = "Failed to open shared memory";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }
  std::unique_ptr<int, decltype(shmDeleter)> shmFd(new int(shm_fd), shmDeleter);

  if (ftruncate(*shmFd.get(), shmSize) < 0) {
    const char* err = "Failed to truncate shared memory";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  /****************************************/
  /* Map the shared memory to the process */
  /****************************************/

  auto munmapDeleter = [shmSize](void* ptr) {
    if (ptr && ptr != MAP_FAILED)
      munmap(ptr, shmSize);
  };
  std::unique_ptr<void, decltype(munmapDeleter)> shmPtr(mmap(nullptr, shmSize, PROT_READ | PROT_WRITE, MAP_SHARED, *shmFd.get(), 0), munmapDeleter);
  if (shmPtr.get() == MAP_FAILED) {
    const char* err = "Failed to map shared memory";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }
  pctx.sharedMem = shmPtr.get();

  /*****************************/
  /* Initialize the semaphores */
  /*****************************/

  #define CAST_SHM(type, offset) reinterpret_cast<type>(reinterpret_cast<char*>(shmPtr.get()) + offset)

  sem_t* childProcessReady = CAST_SHM(sem_t*, 0);
  if (sem_init(childProcessReady, 1, 0) < 0) {
    const char* err = "Failed to initialize child process ready semaphore";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  sem_t* captureSync = CAST_SHM(sem_t*, sizeof(sem_t));
  if (sem_init(captureSync, 1, 1) < 0) {
    const char* err = "Failed to initialize capture sync semaphore";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  /***************************/
  /* Spawn the child process */
  /***************************/

  pid_t childPid;
  char* path = (char*)"/home/afessler/Documents/video_capture/framestream";
  char *argv[] = {path, nullptr};
  int spawnStatus = posix_spawn(&childPid, argv[0], nullptr, nullptr, argv, environ);
  if (spawnStatus != 0) {
    const char* err = "Failed to spawn child process";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return spawnStatus;
  }

  /*****************************/
  /* Pin the process to core 3 */
  /*****************************/

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(3, &cpuset);
  if (sched_setaffinity(0, sizeof(cpuset), &cpuset) < 0) {
    const char* err = "Failed to set CPU affinity";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  /*******************************************************/
  /* Set the scheduling policy to FIFO with max priority */
  /*******************************************************/

  struct sched_param param;
  param.sched_priority = sched_get_priority_max(SCHED_FIFO);
  if (sched_setscheduler(0, SCHED_FIFO, &param) < 0) {
    const char* err = "Failed to set real-time scheduling policy";
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
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
    pctx.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return -errno;
  }

  /***********************************************/
  /* Register the process with the kernel module */
  /***********************************************/

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

  /***************************************************************/
  /* Wait for the child process, then begin and wait for signals */
  /***************************************************************/

  sem_wait(childProcessReady);
  pctx.running = 1;
  while (pctx.running) {
    pause();
  }

  return 0;
}
