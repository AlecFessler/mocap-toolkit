#include <cstring>
#include <semaphore.h>
#include <sys/mman.h>
#include "camera_handler.h"
#include "shared_defs.h"
#include "logger.h"

camera_handler_t::camera_handler_t(
  std::pair<unsigned int, unsigned int> resolution,
  int buffer_count,
  std::pair<std::int64_t, std::int64_t> frame_duration_limits
) : p_ctx_(p_ctx_t::get_instance()), next_req_idx_(0) {

  /*********************************/
  /* Initialize the Camera Manager */
  /*********************************/

  cm_ = std::make_unique<libcamera::CameraManager>();
  if (cm_->start() < 0) {
    const char* err = "Failed to start camera manager";
    p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  /*****************************/
  /* Get the cameras available */
  /*****************************/

  auto cameras = cm_->cameras();
  if (cameras.empty()) {
    const char* err = "No cameras available";
    p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  /***************************************/
  /* Select and acquire the first camera */
  /***************************************/

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

  /*************************************/
  /* Generate the camera configuration */
  /*************************************/

  config_ = camera_->generateConfiguration({ libcamera::StreamRole::VideoRecording });
  if (!config_) {
    const char* err = "Failed to generate camera configuration";
    p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  /*********************************************************/
  /* Adjust the configuration to based on input parameters */
  /*********************************************************/

  libcamera::StreamConfiguration& cfg = config_->at(0);
  cfg.size = { resolution.first, resolution.second };
  cfg.bufferCount = buffer_count;

  /**************************************************/
  /* Validate and possibly adjust the configuration */
  /**************************************************/

  if (config_->validate() == libcamera::CameraConfiguration::Invalid) {
    const char* err = "Invalid camera configuration, unable to adjust";
    p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  } else if (config_->validate() == libcamera::CameraConfiguration::Adjusted) {
    const char* err = "Adjusted invalid camera configuration";
    p_ctx_.logger->log(logger_t::level_t::WARNING, __FILE__, __LINE__, err);
  }

  /***************************/
  /* Apply the configuration */
  /***************************/

  if (camera_->configure(config_.get()) < 0) {
    const char* err = "Failed to configure camera";
    p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  /********************/
  /* Allocate buffers */
  /********************/

  allocator_ = std::make_unique<libcamera::FrameBufferAllocator>(camera_);
  stream_ = cfg.stream();
  if (allocator_->allocate(stream_) < 0) {
    const char* err = "Failed to allocate buffers";
    p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  /********************************************************************/
  /* Create requests, add buffers, and map buffers into shared memory */
  /********************************************************************/

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

    const libcamera::FrameBuffer::Plane& plane = buffer->planes()[0];
    if (IMAGE_BYTES != plane.length) {
      const char* err = "Image size does not match expected size";
      p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
      throw std::runtime_error(err);
    }

    unsigned char* data = reinterpret_cast<unsigned char*>(mmap(
      nullptr,
      plane.length,
      PROT_READ | PROT_WRITE,
      MAP_SHARED,
      plane.fd.get(),
      plane.offset
    ));

    if (data == MAP_FAILED) {
      p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "mmap failed");
      throw std::runtime_error("Failed to mmap plane data");
    }

    mmap_buffers_.push_back(data);
  }

  /**************************************************************************/
  /* Set request complete callback and start camera with specified controls */
  /**************************************************************************/

  camera_->requestCompleted.connect(this,& camera_handler_t::request_complete);
  controls_ = std::make_unique<libcamera::ControlList>();
  controls_->set(libcamera::controls::FrameDurationLimits, libcamera::Span<const std::int64_t, 2>({ frame_duration_limits.first, frame_duration_limits.second }));
  if (camera_->start(controls_.get()) < 0) {
    const char* err = "Failed to start camera";
    p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }
}

camera_handler_t::~camera_handler_t() {
  for (unsigned char* data : mmap_buffers_)
    munmap(data, IMAGE_BYTES);
  camera_->stop();
  allocator_->free(stream_);
  camera_->release();
  camera_.reset();
  cm_->stop();
}

void camera_handler_t::queue_request() {
  /**
   * Queue the next request in the sequence.
   *
   * If requests are not returned at the same rate as they are queued,
   * this method will throw to signal that the camera is not keeping up,
   * and this should be handled by adjusting the configuration.
   * ie. framerate, exposure, gain, etc.
   *
   * Throws:
   *  - std::runtime_error: Failed to queue request
   */
  size_t index = next_req_idx_.load();
  if (camera_->queueRequest(requests_[index].get()) < 0) {
    const char* err = "Failed to queue request";
    p_ctx_.logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }
  next_req_idx_.store((index + 1) % requests_.size());
}

void camera_handler_t::request_complete(libcamera::Request* request) {
  /**
   * Callback for when a request is completed.
   *
   * This method is called by the camera manager when a request is completed.
   * The request is then reused and the image is written to shared memory.
   *
   * The shared memory is structured as follows:
   * - sem_t sem (not used here)
   * - sem_t img_write_sem
   * - sem_t img_read_sem
   * - unsigned char imageBuffer[IMAGE_BYTES]
   *
   * Parameters:
   *  - request: The completed request
   */
  if (request->status() == libcamera::Request::RequestCancelled)
    return;

  const char* info = "Request completed";
  p_ctx_.logger->log(logger_t::level_t::INFO, __FILE__, __LINE__, info);

  static sem_t* img_write_sem = PTR_MATH_CAST(sem_t, p_ctx_.shared_mem, sizeof(sem_t));
  static sem_t* img_read_sem = PTR_MATH_CAST(sem_t, p_ctx_.shared_mem, 2 * sizeof(sem_t));
  static unsigned char* shared_img_buffer = PTR_MATH_CAST(unsigned char, p_ctx_.shared_mem, 3 * sizeof(sem_t));

  unsigned char* data = mmap_buffers_[request->cookie()];
  sem_wait(img_write_sem);
  memcpy(shared_img_buffer, data, IMAGE_BYTES);
  sem_post(img_read_sem);

  request->reuse(libcamera::Request::ReuseBuffers);
}
