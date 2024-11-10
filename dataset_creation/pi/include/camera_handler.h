// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef CAMERAHANDLER_H
#define CAMERAHANDLER_H

#include <cstdint>
#include <semaphore>
#include <memory>
#include <vector>
#include <libcamera/libcamera.h>
#include "config_parser.h"
#include "logger.h"
#include "lock_free_queue.h"

class camera_handler_t {
public:
  camera_handler_t(
    config_parser& config,
    logger_t& logger,
    lock_free_queue_t& frame_queue,
    sem_t& queue_counter
  );
  ~camera_handler_t();
  camera_handler_t(const camera_handler_t&) = delete;
  camera_handler_t& operator=(const camera_handler_t&) = delete;
  camera_handler_t(camera_handler_t&&) = delete;
  camera_handler_t& operator=(camera_handler_t&&) = delete;
  void queue_request();
private:
  logger_t& logger;
  lock_free_queue_t& frame_queue;
  sem_t& queue_counter;
  std::unique_ptr<libcamera::CameraManager> cm_;
  std::shared_ptr<libcamera::Camera> camera_;
  std::unique_ptr<libcamera::CameraConfiguration> config_;
  std::unique_ptr<libcamera::FrameBufferAllocator> allocator_;
  std::unique_ptr<libcamera::ControlList> controls_;
  libcamera::Stream* stream_;
  std::vector<std::unique_ptr<libcamera::Request>> requests_;
  std::vector<void*> mmap_buffers_;
  int next_req_idx_;
  int frame_bytes_;
  int dma_frame_buffers_;
  int frame_buffers_;
  void* frame_bytes_buffer_;
  unsigned int frame_bytes_offset_;
  void request_complete(libcamera::Request* request);
};

#endif // CAMERAHANDLER_H
