// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef CAMERAHANDLER_H
#define CAMERAHANDLER_H

#include <cstdint>
#include <memory>
#include <semaphore.h>
#include <signal.h>
#include <vector>
#include <libcamera/libcamera.h>
#include "config.h"

class camera_handler_t {
public:
  camera_handler_t(
    config& config,
    sem_t& loop_ctl_sem,
    volatile sig_atomic_t& frame_rdy
  );
  ~camera_handler_t();
  camera_handler_t(const camera_handler_t&) = delete;
  camera_handler_t& operator=(const camera_handler_t&) = delete;
  camera_handler_t(camera_handler_t&&) = delete;
  camera_handler_t& operator=(camera_handler_t&&) = delete;
  void queue_request();

  uint8_t* frame_buffer;

private:
  void init_frame_bytes(config& config);
  void init_camera_manager();
  void init_camera_config(config& config);
  void init_dma_buffer();
  void init_camera_controls(config& config);
  void request_complete(libcamera::Request* request);

  size_t frame_bytes_;

  sem_t& loop_ctl_sem;
  volatile sig_atomic_t& frame_rdy;

  std::unique_ptr<libcamera::Request> request;
  std::unique_ptr<libcamera::CameraManager> cm_;
  std::shared_ptr<libcamera::Camera> camera_;
  std::unique_ptr<libcamera::CameraConfiguration> config_;
  std::unique_ptr<libcamera::FrameBufferAllocator> allocator_;
  std::unique_ptr<libcamera::ControlList> controls_;
  libcamera::Stream* stream_;
};

#endif // CAMERAHANDLER_H
