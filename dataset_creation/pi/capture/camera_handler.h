#ifndef CAMERAHANDLER_H
#define CAMERAHANDLER_H

#include <atomic>
#include <memory>
#include <vector>
#include <libcamera/libcamera.h>
#include "p_ctx.h"

class p_ctx_t;

class camera_handler_t {
public:
  camera_handler_t(
    std::pair<unsigned int, unsigned int> resolution,
    int buffer_count,
    std::pair<std::int64_t, std::int64_t> frame_duration_limits
  );
  ~camera_handler_t();
  camera_handler_t(const camera_handler_t&) = delete;
  camera_handler_t& operator=(const camera_handler_t&) = delete;
  camera_handler_t(camera_handler_t&&) = delete;
  camera_handler_t& operator=(camera_handler_t&&) = delete;
  void queue_request();
private:
  p_ctx_t& p_ctx_;
  std::unique_ptr<libcamera::CameraManager> cm_;
  std::shared_ptr<libcamera::Camera> camera_;
  std::unique_ptr<libcamera::CameraConfiguration> config_;
  std::unique_ptr<libcamera::FrameBufferAllocator> allocator_;
  std::unique_ptr<libcamera::ControlList> controls_;
  libcamera::Stream* stream_;
  std::vector<std::unique_ptr<libcamera::Request>> requests_;
  std::vector<void*> mmap_buffers_;
  std::atomic<size_t> next_req_idx_;
  void request_complete(libcamera::Request* request);
};

#endif // CAMERAHANDLER_H
