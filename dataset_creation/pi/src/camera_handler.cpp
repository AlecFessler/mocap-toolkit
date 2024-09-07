// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <cstring>
#include <semaphore.h>
#include <sys/mman.h>
#include "camera_handler.h"
#include "logger.h"

#include <iostream>

camera_handler_t::camera_handler_t(
  std::pair<unsigned int, unsigned int> resolution,
  int buffer_count,
  std::pair<std::int64_t, std::int64_t> frame_duration_limits
) :
  p_ctx_(p_ctx_t::get_instance()),
  next_req_idx_(0) {
  cm_ = std::make_unique<libcamera::CameraManager>();
  if (cm_->start() < 0) {
    const char* err = "Failed to start camera manager";
    p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  auto cameras = cm_->cameras();
  if (cameras.empty()) {
    const char* err = "No cameras available";
    p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  camera_ = cm_->get(cameras[0]->id());
  if (!camera_) {
    const char* err = "Failed to retrieve camera";
    p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }
  if (camera_->acquire() < 0) {
    const char* err = "Failed to acquire camera";
    p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  config_ = camera_->generateConfiguration({ libcamera::StreamRole::VideoRecording });
  if (!config_) {
    const char* err = "Failed to generate camera configuration";
    p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  libcamera::StreamConfiguration& cfg = config_->at(0);
  cfg.size = { resolution.first, resolution.second };
  cfg.bufferCount = buffer_count;

  if (config_->validate() == libcamera::CameraConfiguration::Invalid) {
    const char* err = "Invalid camera configuration, unable to adjust";
    p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  } else if (config_->validate() == libcamera::CameraConfiguration::Adjusted) {
    const char* err = "Invalid camera configuration, adjusted";
    p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  if (camera_->configure(config_.get()) < 0) {
    const char* err = "Failed to configure camera";
    p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  allocator_ = std::make_unique<libcamera::FrameBufferAllocator>(camera_);
  stream_ = cfg.stream();
  if (allocator_->allocate(stream_) < 0) {
    const char* err = "Failed to allocate buffers";
    p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  uint64_t req_cookie = 0; // maps request to index in mmap_buffers_
  for (const std::unique_ptr<libcamera::FrameBuffer>& buffer : allocator_->buffers(stream_)) {
    std::unique_ptr<libcamera::Request> request = camera_->createRequest(req_cookie++);
    if (!request) {
      const char* err = "Failed to create request";
      p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
      throw std::runtime_error(err);
    }
    if (request->addBuffer(stream_, buffer.get()) < 0) {
      const char* err = "Failed to add buffer to request";
      p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
      throw std::runtime_error(err);
    }
    requests_.push_back(std::move(request));

    const libcamera::FrameBuffer::Plane& y_plane = buffer->planes()[0];
    const libcamera::FrameBuffer::Plane& u_plane = buffer->planes()[1];
    const libcamera::FrameBuffer::Plane& v_plane = buffer->planes()[2];

    if (y_plane.length != Y_PLANE || u_plane.length != UV_PLANE || v_plane.length != UV_PLANE) {
      const char* err = "Plane size does not match expected size";
      p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
      throw std::runtime_error(err);
    }

    void* data = mmap(
      nullptr,
      IMAGE_BYTES,
      PROT_READ | PROT_WRITE,
      MAP_SHARED,
      y_plane.fd.get(),
      y_plane.offset
    );

    if (data == MAP_FAILED) {
      std::string err = "Failed to mmap plane data: " + std::string(strerror(errno));
      p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err.c_str());
      throw std::runtime_error(err);
    }

    mmap_buffers_.push_back(data);
  }

  camera_->requestCompleted.connect(this,& camera_handler_t::request_complete);

  // Configure some settings for more deterministic capture times
  // May need to be adjusted based on lighting conditions
  // and on a per device basis, but for development purposes, this is acceptable
  controls_ = std::make_unique<libcamera::ControlList>();

  // Fix exposure time to half the time between frames
  // May be able to remove frame duration limit control since we are setting exposure
  controls_->set(libcamera::controls::FrameDurationLimits, libcamera::Span<const std::int64_t, 2>({ frame_duration_limits.first, frame_duration_limits.second }));
  controls_->set(libcamera::controls::AeEnable, false);
  controls_->set(libcamera::controls::ExposureTime, frame_duration_limits.first);

  // Fix focus to ~12 inches
  // Focus value should be reciprocal of distance in meters
  controls_->set(libcamera::controls::AfMode, libcamera::controls::AfModeManual);
  controls_->set(libcamera::controls::LensPosition, 3.33);

  // Fix white balance, gain, and disable HDR
  controls_->set(libcamera::controls::AwbEnable, false);
  controls_->set(libcamera::controls::AnalogueGain, 1.0);
  controls_->set(libcamera::controls::HdrMode, libcamera::controls::HdrModeOff);

  controls_->set(libcamera::controls::rpi::StatsOutputEnable, false);

  if (camera_->start(controls_.get()) < 0) {
    const char* err = "Failed to start camera";
    p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }
}

camera_handler_t::~camera_handler_t() {
  camera_->stop();
  for (void* data : mmap_buffers_)
    munmap(data, IMAGE_BYTES);
  allocator_->free(stream_);
  allocator_.reset();
  camera_->release();
  camera_.reset();
  cm_->stop();
}

void camera_handler_t::queue_request() {
  /**
   * Queue the next request in the sequence.
   *
   * Before queuing the request, ensure that the number of enqueued
   * buffers is no more than PREALLOCATED_BUFFERS - 2. This is because
   * the queue counter may fall behind by, but no more than, 1. This
   * occurs when the child thread calls sem_wait, decrementing the
   * semaphore, but before it dequeues the buffer. Thus, we check
   * for 2 less than max to ensure at least one is available even
   * if the counter is behind by 1.
   *
   * If requests are not returned at the same rate as they are queued,
   * this method will throw to signal that the camera is not keeping up,
   * and this should be handled by adjusting the configuration.
   * ie. framerate, exposure, gain, etc.
   *
   * Throws:
   *  - std::runtime_error: Buffer is not ready for requeuing
   *  - std::runtime_error: Failed to queue request
   */
  int enqueued_buffers = 0;
  sem_getvalue(p_ctx_.queue_counter, &enqueued_buffers);
  if ((unsigned int)enqueued_buffers > PREALLOCATED_BUFFERS - 2) {
    const char* err = "Buffer is not ready for requeuing";
    p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }
  if (camera_->queueRequest(requests_[next_req_idx_].get()) < 0) {
    const char* err = "Failed to queue request";
    p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }
  ++next_req_idx_;
  next_req_idx_ %= requests_.size();
}

void camera_handler_t::request_complete(libcamera::Request* request) {
  /**
   * Signal handler for when a request is completed.
   *
   * This method is called by the camera manager when a request is completed.
   * The request is then reused and the buffer is enqueued for transmission.
   * The queue counter is incremented to signal that a buffer is available.
   *
   * Parameters:
   *  - request: The completed request
   */
  if (request->status() == libcamera::Request::RequestCancelled)
    return;

  const char* info = "Request completed";
  p_ctx_.logger->log(logger_t::level_t::INFO, __FILE__, __LINE__, info);

  void* data = mmap_buffers_[request->cookie()];
  bool enqueued = false;
  do {
    enqueued = p_ctx_.frame_queue->enqueue(data);
  } while(!enqueued);

  sem_post(p_ctx_.queue_counter);
  request->reuse(libcamera::Request::ReuseBuffers);
}
