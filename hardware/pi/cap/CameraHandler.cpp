#include <semaphore.h>
#include "CameraHandler.h"
#include "Logger.h"

using namespace libcamera;

CameraHandler::CameraHandler(
  std::pair<unsigned int, unsigned int> resolution,
  int bufferCount,
  std::pair<std::int64_t, std::int64_t> frameDurationLimits
) : pctx_(PCtx::getInstance()), nextRequestIndex_(0) {

  /*********************************/
  /* Initialize the Camera Manager */
  /*********************************/

  cm_ = std::make_unique<CameraManager>();
  if (cm_->start() < 0) {
    static const char* err = "Failed to start camera manager";
    pctx_.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  /*****************************/
  /* Get the cameras available */
  /*****************************/

  auto cameras = cm_->cameras();
  if (cameras.empty()) {
    static const char* err = "No cameras available";
    pctx_.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  /***************************************/
  /* Select and acquire the first camera */
  /***************************************/

  camera_ = cm_->get(cameras[0]->id());
  if (!camera_) {
    static const char* err = "Failed to retrieve camera";
    pctx_.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }
  if (camera_->acquire() < 0) {
    static const char* err = "Failed to acquire camera";
    pctx_.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  /*************************************/
  /* Generate the camera configuration */
  /*************************************/

  config_ = camera_->generateConfiguration({ StreamRole::VideoRecording });
  if (!config_) {
    static const char* err = "Failed to generate camera configuration";
    pctx_.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  /*********************************************************/
  /* Adjust the configuration to based on input parameters */
  /*********************************************************/

  StreamConfiguration &cfg = config_->at(0);
  cfg.size = { resolution.first, resolution.second };
  cfg.bufferCount = bufferCount;

  /**************************************************/
  /* Validate and possibly adjust the configuration */
  /**************************************************/

  if (config_->validate() == CameraConfiguration::Invalid) {
    static const char* err = "Invalid camera configuration, unable to adjust";
    pctx_.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  } else if (config_->validate() == CameraConfiguration::Adjusted) {
    static const char* err = "Adjusted invalid camera configuration";
    pctx_.logger->log(Logger::Level::WARNING, __FILE__, __LINE__, err);
  }

  /***************************/
  /* Apply the configuration */
  /***************************/

  if (camera_->configure(config_.get()) < 0) {
    static const char* err = "Failed to configure camera";
    pctx_.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  /********************/
  /* Allocate buffers */
  /********************/

  allocator_ = std::make_unique<FrameBufferAllocator>(camera_);
  stream_ = cfg.stream();
  if (allocator_->allocate(stream_) < 0) {
    static const char* err = "Failed to allocate buffers";
    pctx_.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  /**************************************/
  /* Create requests and assign buffers */
  /**************************************/

  for (const std::unique_ptr<FrameBuffer> &buffer : allocator_->buffers(stream_)) {
    std::unique_ptr<Request> request = camera_->createRequest();
    if (!request) {
      static const char* err = "Failed to create request";
      pctx_.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
      throw std::runtime_error(err);
    }
    if (request->addBuffer(stream_, buffer.get()) < 0) {
      static const char* err = "Failed to add buffer to request";
      pctx_.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
      throw std::runtime_error(err);
    }
    requests_.push_back(std::move(request));
  }

  /**************************************************************************/
  /* Set request complete callback and start camera with specified controls */
  /**************************************************************************/

  camera_->requestCompleted.connect(this, &CameraHandler::requestComplete);
  controls_ = std::make_unique<ControlList>();
  controls_->set(controls::FrameDurationLimits, Span<const std::int64_t, 2>({ frameDurationLimits.first, frameDurationLimits.second }));
  if (camera_->start(controls_.get()) < 0) {
    static const char* err = "Failed to start camera";
    pctx_.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }
}

CameraHandler::~CameraHandler() {
  camera_->stop();
  allocator_->free(stream_);
  camera_->release();
  camera_.reset();
  cm_->stop();
}

void CameraHandler::queueRequest() {
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
  size_t index = nextRequestIndex_.load();
  if (camera_->queueRequest(requests_[index].get()) < 0) {
    static const char* err = "Failed to queue request";
    pctx_.logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }
  nextRequestIndex_.store((index + 1) % requests_.size());
}

void CameraHandler::requestComplete(Request *request) {
  /**
   * Callback for when a request is completed.
   *
   * This method is called by the camera manager when a request is completed.
   * The request is then reused and the image is written to shared memory.
   *
   * The shared memory is structured as follows:
   * - sem_t sem (not used here)
   * - sem_t sem (synchronizes access to shared memory with child process)
   * - unsigned char imageBuffer[IMAGE_SIZE]
   *
   * Parameters:
   *  - request: The completed request
   */
  if (request->status() == Request::RequestCancelled)
    return;
  #define CAST_SHM(type, offset) reinterpret_cast<type>(reinterpret_cast<char*>(pctx_.sharedMem) + offset)
  static sem_t* sem = CAST_SHM(sem_t*, sizeof(sem_t));
  static unsigned char* imageBuffer = CAST_SHM(unsigned char*, sizeof(sem_t) * 2);
  const char* info = "Request completed";
  pctx_.logger->log(Logger::Level::INFO, __FILE__, __LINE__, info);
  sem_wait(sem);
  // Write image to shared memory
  sem_post(sem);

  request->reuse(Request::ReuseBuffers);
}
