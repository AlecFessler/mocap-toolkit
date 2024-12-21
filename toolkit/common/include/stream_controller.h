#ifndef STREAM_CONTROLLER_H
#define STREAM_CONTROLLER_H

#include <cstddef>
#include <opencv2/core.hpp>
#include <semaphore.h>
#include <sys/types.h>

#define SHM_NAME "/mocap-toolkit_frameset"
#define SEM_NAME "/mocap-toolkit_consumer_ready"
#define SERVER_EXE "/usr/local/bin/mocap-toolkit-server"

class StreamController {
private:
  size_t frame_width;
  size_t frame_height;
  size_t num_cameras;
  pid_t server_pid_;
  sem_t* frameset_ctl_sem;
  int shm_fd;
  size_t shm_size;
  void* frameset_buf;

public:
  StreamController(
    size_t frame_width,
    size_t frame_height,
    size_t num_cameras
  );
  ~StreamController();

  void recv_frameset(cv::Mat* frames, uint64_t* timestamp);

  StreamController(const StreamController&) = delete;
  StreamController& operator=(const StreamController&) = delete;
  StreamController(StreamController&&) = delete;
  StreamController& operator=(StreamController&&) = delete;
};

#endif // STREAM_CONTROLLER_H
