// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <array>
#include <libcamera/libcamera.h>
#include <memory>
#include <span>

constexpr uint32_t NUM_DMA_BUFS = 2;

struct req_buffer {
  std::unique_ptr<libcamera::Request> request;
  std::span<uint8_t> buffer;
};

class Camera {
private:
  std::unique_ptr<libcamera::CameraManager> m_cam_mgr;
  std::shared_ptr<libcamera::Camera> m_cam;
  std::unique_ptr<libcamera::FrameBufferAllocator> m_alloc;
  libcamera::Stream* m_stream;
  std::array<req_buffer, NUM_DMA_BUFS> m_buffers;

public:
  Camera(
    std::pair<uint32_t, uint32_t> resolution,
    uint32_t fps
  );
  ~Camera();
  void capture_frame();
  void request_complete(libcamera::Request* request);
};

#endif // CAMERA_HPP
