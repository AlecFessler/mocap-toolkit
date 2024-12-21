#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <errno.h>
#include <opencv2/core.hpp>
#include <semaphore.h>
#include <stdexcept>
#include <string.h>
#include <system_error>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "logging.h"
#include "stream_controller.h"

StreamController::StreamController(
  size_t frame_width,
  size_t frame_height,
  size_t num_cameras
) :
  frame_width(frame_width),
  frame_height(frame_height),
  num_cameras(num_cameras),
  server_pid_(0),
  frameset_ctl_sem(nullptr),
  shm_fd(-1),
  frameset_buf(nullptr)
{
  char logstr[128];

  server_pid_ = fork();
  if (server_pid_ == -1) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Failed to fork process: %s",
      strerror(errno)
    );
    LOG(ERROR, logstr);
    throw std::runtime_error(logstr);
  }

  if (server_pid_ == 0) {
    execl(SERVER_EXE, SERVER_EXE, nullptr);
    _exit(errno);
  }

  frameset_ctl_sem = sem_open(
    SEM_NAME,
    O_CREAT,
    0666,
    0
  );
  if (frameset_ctl_sem == SEM_FAILED) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error opening semaphore: %s",
      strerror(errno)
    );
    LOG(ERROR, logstr);
    throw std::runtime_error(logstr);
  }

  shm_fd = shm_open(
    SHM_NAME,
    O_CREAT | O_RDWR,
    0666
  );
  if (shm_fd == -1) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error creating shared memory: %s",
      strerror(errno)
    );
    LOG(ERROR, logstr);
    throw std::runtime_error(logstr);
  }

  shm_size = frame_width * frame_height * 3 / 2 * num_cameras + sizeof(uint64_t);

  int ret = ftruncate(
    shm_fd,
    shm_size
  );
  if (ret == -1) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error resizing shared memory: %s",
      strerror(errno)
    );
    LOG(ERROR, logstr);
    throw std::runtime_error(logstr);
  }

  frameset_buf = mmap(
    NULL,
    shm_size,
    PROT_READ,
    MAP_SHARED,
    shm_fd,
    0
  );
  if (frameset_buf == MAP_FAILED) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error mapping shared memory: %s",
      strerror(errno)
    );
    LOG(ERROR, logstr);
    throw std::runtime_error(logstr);
  }
}

StreamController::~StreamController() {
  if (server_pid_ > 0)
    kill(server_pid_, SIGTERM);

  if (frameset_buf != nullptr)
    munmap(frameset_buf, shm_size);

  if (shm_fd > -1) {
    close(shm_fd);
    shm_unlink(SHM_NAME);
  }

  if (frameset_ctl_sem != nullptr) {
    sem_close(frameset_ctl_sem);
    sem_unlink(SEM_NAME);
  }
}

void StreamController::recv_frameset(cv::Mat* frames, uint64_t* timestamp) {
  sem_wait(frameset_ctl_sem);

  size_t frame_size = frame_width * frame_height * 3 / 2;
  for(size_t i = 0; i < num_cameras; i++) {
    size_t offset = i * frame_size;
    uint8_t* frame = static_cast<uint8_t*>(frameset_buf) + offset;

    frames[i] = cv::Mat(
      frame_height * 3/2,
      frame_width,
      CV_8UC1,
      frame
    ).clone();
  }

  uint8_t* timestamp_ptr = static_cast<uint8_t*>(frameset_buf) + (frame_size * num_cameras);
  *timestamp = *reinterpret_cast<uint64_t*>(timestamp_ptr);
}
