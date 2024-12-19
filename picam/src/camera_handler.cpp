#include <cstring>
#include <iostream>
#include <semaphore.h>
#include <stdexcept>
#include <sys/mman.h>

#include "camera_handler.h"
#include "config.h"
#include "logging.h"


camera_handler_t::camera_handler_t(
  config& config,
  sem_t& loop_ctl_sem,
  volatile sig_atomic_t& frame_rdy
) :
  loop_ctl_sem(loop_ctl_sem),
  frame_rdy(frame_rdy) {
  /**
   * Manages camera operations using the libcamera API, providing a high-level interface
   * for frame capture and buffer management. The handler coordinates three key tasks:
   *
   * 1. Camera initialization and configuration
   * 2. DMA buffer management for zero-copy frame capture
   * 3. Frame completion notification via callback system
   *
   * When a frame is captured, libcamera writes directly to a DMA buffer and invokes
   * our callback. The callback then enqueues the buffer pointer to a lock-free queue
   * and signals the main loop via semaphore that a new frame is ready for processing.
   *
   * Parameters:
   *   config:        Camera and frame settings including resolution and buffer counts
   *   frame_queue:   Lock-free queue for sharing frame buffers with main loop
   *   loop_ctl_sem: Semaphore tracking available frames in the queue
   *
   * The initialization sequence is:
   * 1. Configure frame properties (resolution, format)
   * 2. Initialize camera manager and acquire device
   * 3. Apply camera configuration
   * 4. Set up DMA buffers and memory mapping
   * 5. Configure camera controls (exposure, focus, etc)
   */
  init_frame_bytes(config);
  init_camera_manager();
  init_camera_config(config);
  init_dma_buffer();
  init_camera_controls(config);
}

void camera_handler_t::init_frame_bytes(config& config) {
  /**
   * Calculates and stores frame buffer parameters based on YUV420 format.
   *
   * YUV420 format has three planes:
   * - Y (luma): Full resolution (width × height)
   * - U, V (chroma): Quarter resolution (width/2 × height/2 each)
   *
   * Parameters:
   *   config: Contains frame dimensions and buffer count settings
   */
  unsigned int y_plane_bytes = config.frame_width * config.frame_height;
  unsigned int u_plane_bytes = y_plane_bytes / 4;
  unsigned int v_plane_bytes = u_plane_bytes;
  frame_bytes_ = y_plane_bytes + u_plane_bytes + v_plane_bytes;
}

void camera_handler_t::init_camera_manager() {
  /**
   * Initializes and acquires exclusive access to the camera device.
   *
   * The initialization sequence is:
   * 1. Start the camera manager service
   * 2. Enumerate available cameras
   * 3. Select and acquire first available camera
   *
   * Throws:
   *   std::runtime_error: On any initialization failure (detailed in error message)
   */
  cm_ = std::make_unique<libcamera::CameraManager>();
  if (cm_->start() < 0) {
    const char* err = "Failed to start camera manager";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }

  auto cameras = cm_->cameras();
  if (cameras.empty()) {
    const char* err = "No cameras available";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }

  camera_ = cm_->get(cameras[0]->id());
  if (!camera_) {
    const char* err = "Failed to retrieve camera";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }
  if (camera_->acquire() < 0) {
    const char* err = "Failed to acquire camera";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }
}

void camera_handler_t::init_camera_config(config& config) {
  /**
   * Configures the camera stream for video recording.
   *
   * Sets core recording parameters:
   * - YUV420 pixel format for efficient encoding
   * - Frame resolution from config
   * - Number of DMA buffers to allocate
   *
   * The configuration is validated to ensure the camera supports these settings
   * without requiring adjustments.
   *
   * Parameters:
   *   config: Contains frame dimensions and buffer settings
   *
   * Throws:
   *   std::runtime_error: If configuration is invalid or fails to apply
   */
  config_ = camera_->generateConfiguration({ libcamera::StreamRole::VideoRecording });
  if (!config_) {
    const char* err = "Failed to generate camera configuration";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }

  libcamera::StreamConfiguration& cfg = config_->at(0);
  cfg.pixelFormat = libcamera::formats::YUV420;
  cfg.size = { (unsigned int)config.frame_width, (unsigned int)config.frame_height };
  cfg.bufferCount = 1;

  if (config_->validate() == libcamera::CameraConfiguration::Invalid) {
    const char* err = "Invalid camera configuration, unable to adjust";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  } else if (config_->validate() == libcamera::CameraConfiguration::Adjusted) {
    const char* err = "Invalid camera configuration, adjusted";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }

  if (camera_->configure(config_.get()) < 0) {
    const char* err = "Failed to configure camera";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }
}

void camera_handler_t::init_dma_buffer() {
  allocator_ = std::make_unique<libcamera::FrameBufferAllocator>(camera_);
  stream_ = config_->at(0).stream();
  if (allocator_->allocate(stream_) < 0) {
    const char* err = "Failed to allocate buffers";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }

  const std::unique_ptr<libcamera::FrameBuffer>& buffer = allocator_->buffers(stream_)[0];
  request = camera_->createRequest();
  if (!request) {
    const char* err = "Failed to create request";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }

  if (request->addBuffer(stream_, buffer.get()) < 0) {
    const char* err = "Failed to add buffer to request";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }

  const libcamera::FrameBuffer::Plane& y_plane = buffer->planes()[0];
  const libcamera::FrameBuffer::Plane& u_plane = buffer->planes()[1];
  const libcamera::FrameBuffer::Plane& v_plane = buffer->planes()[2];

  unsigned int y_plane_bytes = frame_bytes_ * 2/3;  // Based on YUV420
  unsigned int u_plane_bytes = y_plane_bytes / 4;
  unsigned int v_plane_bytes = u_plane_bytes;

  if (y_plane.length != y_plane_bytes || u_plane.length != u_plane_bytes || v_plane.length != v_plane_bytes) {
    const char* err = "Plane size does not match expected size";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }

  void* data = mmap(
    nullptr,
    frame_bytes_,
    PROT_READ | PROT_WRITE,
    MAP_SHARED,
    y_plane.fd.get(),
    y_plane.offset
  );

  if (data == MAP_FAILED) {
    char logstr[128];
    snprintf(
      logstr,
      sizeof(logstr),
      "Failed to mmap plane data: %s",
      strerror(errno)
    );
    LOG(ERROR, logstr);
    throw std::runtime_error(logstr);
  }

  frame_buffer = (uint8_t*)data;

  camera_->requestCompleted.connect(this, &camera_handler_t::request_complete);
}

void camera_handler_t::init_camera_controls(config& config) {
  /**
   * Configures camera settings for consistent, high-quality video capture.
   *
   * Following cinematography best practices, this method:
   * 1. Sets exposure time to half the frame interval (180° shutter rule)
   *    This prevents motion blur while maintaining natural motion appearance
   *
   * 2. Disables automatic controls that could cause frame timing variation:
   *    - Auto exposure (AE)
   *    - Auto white balance (AWB)
   *    - Auto focus (AF)
   *    - HDR mode
   *
   * 3. Fixes focus at ~12 inches (lens position 3.33)
   *    The lens position is the reciprocal of the focus distance in meters
   *
   * 4. Sets gain to 1.0 (equivalent to ISO 100) for minimal noise
   *
   * Parameters:
   *   config: Contains frame timing constraints (duration_min/max)
   *
   * Throws:
   *   std::runtime_error if camera fails to start with these settings
   *
   * Note:
   *   These parameters will change, but are fine for development
   */
  unsigned int frame_duration_min = config.frame_duration_min;
  unsigned int frame_duration_max = config.frame_duration_max;

  controls_ = std::make_unique<libcamera::ControlList>();

  // Fix exposure time to half the time between frames
  // May be able to remove frame duration limit control since we are setting exposure
  controls_->set(libcamera::controls::FrameDurationLimits, libcamera::Span<const std::int64_t, 2>({ frame_duration_min, frame_duration_max }));
  controls_->set(libcamera::controls::AeEnable, false);
  controls_->set(libcamera::controls::ExposureTime, frame_duration_min);

  if (camera_->start(controls_.get()) < 0) {
    const char* err = "Failed to start camera";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }
}

camera_handler_t::~camera_handler_t() {
  /**
   * Cleanly shuts down camera and releases resources.
   *
   * Strict cleanup order is required:
   * 1. Stop camera capture
   * 2. Unmap DMA buffers
   * 3. Free buffer allocator
   * 4. Release camera device
   * 5. Stop camera manager
   *
   * Warning: Do not modify this sequence as it may cause
   * undefined behavior or resource leaks
   */
  camera_->stop();
  munmap(frame_buffer, frame_bytes_);
  allocator_->free(stream_);
  allocator_.reset();
  camera_->release();
  camera_.reset();
  cm_->stop();
}

void camera_handler_t::queue_request() {
  if (camera_->queueRequest(request.get()) < 0) {
    const char* err = "Failed to queue request";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }
}

void camera_handler_t::request_complete(libcamera::Request* request) {
  if (request->status() == libcamera::Request::RequestCancelled)
    return;

  request->reuse(libcamera::Request::ReuseBuffers);
  frame_rdy = 1;
  sem_post(&loop_ctl_sem);
}
