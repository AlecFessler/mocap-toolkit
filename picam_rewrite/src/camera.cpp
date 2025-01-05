// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <cerrno>
#include <chrono>
#include <cstring>
#include <span>
#include <string>
#include <sys/mman.h>
#include <libcamera/libcamera.h>
#include <memory>

#include "camera.hpp"
#include "logging.hpp"
#include "scheduling.hpp"
#include "worker_threads.hpp"

Camera::Camera(
  std::pair<uint32_t, uint32_t> resolution,
  uint32_t fps,
  WorkerThreads&& workers
) :
  m_workers(std::move(workers)),
  m_thread_setup(false) {

  m_cam_mgr = std::make_unique<libcamera::CameraManager>();
  int status = m_cam_mgr->start();
  if (status < 0) {
    std::string err_msg =
      "Failed to start camera manager: "
      + std::string(strerror(status));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  auto cameras = m_cam_mgr->cameras();
  if (cameras.empty()) {
    std::string err_msg = "No cameras available";
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  m_cam = m_cam_mgr->get(cameras[0]->id());
  status = m_cam->acquire();
  if (status < 0) {
    std::string err_msg =
      "Failed to acquire camera: "
      + std::string(strerror(errno));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  std::unique_ptr<libcamera::CameraConfiguration> conf =
    m_cam->generateConfiguration(
      {libcamera::StreamRole::VideoRecording}
    );
  if (conf == nullptr) {
    std::string err_msg = "Failed to generate camera config";
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  libcamera::StreamConfiguration& stream_conf = conf->at(0);
  stream_conf.pixelFormat = libcamera::formats::YUV420;
  stream_conf.size = {resolution.first, resolution.second};
  stream_conf.bufferCount = NUM_DMA_BUFS;

  auto validation_result = conf->validate();
  if (
    (validation_result == libcamera::CameraConfiguration::Invalid) ||
    (validation_result == libcamera::CameraConfiguration::Adjusted)
  ) {
    std::string err_msg = "Invalid camera configuration";
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  status = m_cam->configure(conf.get());
  if (status < 0) {
    std::string err_msg =
      "Failed to configure camera: "
      + std::string(strerror(status));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  m_alloc = std::make_unique<libcamera::FrameBufferAllocator>(m_cam);
  m_stream = conf->at(0).stream();
  status = m_alloc->allocate(m_stream);
  if (status < 0) {
    std::string err_msg =
      "Failed to allocate frame buffers: "
      + std::string(strerror(status));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  for (uint32_t i = 0; i < NUM_DMA_BUFS; i++) {
    const std::unique_ptr<libcamera::FrameBuffer>& buffer = m_alloc->buffers(m_stream)[i];
    std::unique_ptr<libcamera::Request> req = m_cam->createRequest();
    if (req == nullptr) {
      std::string err_msg = "Failed to create request";
      log_(ERROR, err_msg.c_str());
      throw std::runtime_error(err_msg);
    }

    status = req->addBuffer(m_stream, buffer.get());
    if (status < 0) {
      std::string err_msg =
        "Failed to add buffer to request: "
        + std::string(strerror(errno));
      log_(ERROR, err_msg.c_str());
      throw std::runtime_error(err_msg);
    }

    uint64_t frame_bytes = resolution.first * resolution.second * 3 / 2;
    const libcamera::FrameBuffer::Plane& y_plane = buffer->planes()[0];
    void* dma_buffer = mmap(
      nullptr,
      frame_bytes,
      PROT_READ | PROT_WRITE,
      MAP_SHARED,
      y_plane.fd.get(),
      y_plane.offset
    );
    if (dma_buffer == MAP_FAILED) {
      std::string err_msg =
        "Failed to map DMA buffer: "
        + std::string(strerror(errno));
      log_(ERROR, err_msg.c_str());
      throw std::runtime_error(err_msg);
    }

    struct req_buffer req_buf{
      std::move(req),
      std::chrono::nanoseconds{0},
      std::span<uint8_t>(
        static_cast<uint8_t*>(dma_buffer),
        frame_bytes
      )
    };
    m_buffers[i] = std::move(req_buf);
  }

  m_cam->requestCompleted.connect(this, &Camera::request_complete);

  auto frame_duration = std::chrono::microseconds{std::chrono::seconds{1}} / fps / 2;
  auto controls = std::make_unique<libcamera::ControlList>();
  controls->set(
    libcamera::controls::FrameDurationLimits,
    libcamera::Span<const int64_t, 2>(
      {frame_duration.count(), frame_duration.count()}
    )
  );
  controls->set(
    libcamera::controls::AeEnable,
    false
  );
  controls->set(
    libcamera::controls::ExposureTime,
    frame_duration.count()
  );

  status = m_cam->start(controls.get());
  if (status < 0) {
    std::string err_msg =
      "Failed to start camera: "
      + std::string(strerror(status));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  m_workers.launch_workers();
}

Camera::~Camera() {
  m_cam->stop();
  for (auto& buffer : m_buffers) {
    munmap(
      buffer.buffer.data(),
      buffer.buffer.size_bytes()
    );
  }
  m_alloc->free(m_stream);
  m_alloc.reset();
  m_cam->release();
  m_cam.reset();
  m_cam_mgr->stop();
}

void Camera::capture_frame(std::chrono::nanoseconds timestamp) {
  m_buffers[0].timestamp = timestamp;
  int status = m_cam->queueRequest(m_buffers[0].request.get());
  if (status < 0) {
    std::string err_msg =
      "Failed to queue request: "
      + std::string(strerror(status));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  //log_(BENCHMARK, "Queued capture request");

  std::swap(
    m_buffers[0],
    m_buffers[1]
  );
}

void Camera::request_complete(libcamera::Request* request) {
  if (!m_thread_setup) {
    // maybe a little hacky, but we're using the callback
    // to pin the libcamera camera manager thread to a specific
    // core and set it's scheduling priority. This is because
    // the initial image preprocessing stages run in this thread
    m_thread_setup = true;
    pin_to_core(0);
    set_scheduling_prio(98);
  }

  if (request->status() == libcamera::Request::RequestCancelled)
    return;
  request->reuse(libcamera::Request::ReuseBuffers);

  //log_(BENCHMARK, "Capture request complete");

  std::chrono::nanoseconds timestamp = m_buffers[1].timestamp;
  std::span<uint8_t> frame = m_buffers[1].buffer;
  m_workers.start_task(timestamp, frame);
}
