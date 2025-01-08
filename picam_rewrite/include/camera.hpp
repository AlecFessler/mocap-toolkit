// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <array>
#include <chrono>
#include <libcamera/libcamera.h>
#include <memory>
#include <span>

#include "frame_buffer.hpp"
#include "spsc_queue_wrapper.hpp"

class Camera {
private:
  std::unique_ptr<libcamera::CameraManager> m_cam_mgr;
  std::shared_ptr<libcamera::Camera> m_cam;
  std::unique_ptr<libcamera::FrameBufferAllocator> m_alloc;
  libcamera::Stream* m_stream;
  std::vector<std::unique_ptr<libcamera::Request>> m_reqs;

  uint32_t m_next_buffer;
  std::vector<struct frame> m_buffers;
  SPSCQueue<struct frame>& m_frame_queue;

  bool m_thread_setup;

public:
  Camera(
    std::pair<uint32_t, uint32_t> resolution,
    uint32_t fps,
    uint32_t num_dma_buffers,
    SPSCQueue<struct frame>& frame_queue
  );
  ~Camera();
  void capture_frame(std::chrono::nanoseconds timestamp);
  void request_complete(libcamera::Request* request);
};

#endif // CAMERA_HPP
