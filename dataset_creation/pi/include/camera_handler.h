// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef CAMERAHANDLER_H
#define CAMERAHANDLER_H

#include <cstdint>
#include <memory>
#include <vector>
#include <libcamera/libcamera.h>
#include "config_parser.h"
#include "logger.h"

class camera_handler_t {
public:
  camera_handler_t(config_parser& config, logger_t& logger);
  ~camera_handler_t();
  camera_handler_t(const camera_handler_t&) = delete;
  camera_handler_t& operator=(const camera_handler_t&) = delete;
  camera_handler_t(camera_handler_t&&) = delete;
  camera_handler_t& operator=(camera_handler_t&&) = delete;
  void queue_request();
private:
  config_parser& config;
  logger_t& logger;
  std::unique_ptr<libcamera::CameraManager> cm_;
  std::shared_ptr<libcamera::Camera> camera_;
  std::unique_ptr<libcamera::CameraConfiguration> config_;
  std::unique_ptr<libcamera::FrameBufferAllocator> allocator_;
  std::unique_ptr<libcamera::ControlList> controls_;
  libcamera::Stream* stream_;
  std::vector<std::unique_ptr<libcamera::Request>> requests_;
  std::vector<void*> mmap_buffers_;
  uint64_t next_req_idx_;
  uint32_t y_plane_bytes_;
  uint32_t u_plane_bytes_;
  uint32_t v_plane_bytes_;
  uint32_t frame_bytes_;
  FILE* ffmpeg_;
  void request_complete(libcamera::Request* request);
};

#endif // CAMERAHANDLER_H
