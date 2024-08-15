#include "CameraHandler.h"
#include "Logger.h"

#include <string>

using namespace libcamera;

CameraHandler::CameraHandler(
  std::pair<unsigned int, unsigned int> resolution,
  int bufferCount,
  std::pair<std::int64_t, std::int64_t> frameDurationLimits
) {
  cm_ = std::make_unique<CameraManager>();
  if (cm_->start() < 0) {
    std::string err = "Failed to start camera manager";
    Logger::getLogger().log(Logger::timestamp(), Logger::Level::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  auto cameras = cm_->cameras();
  if (cameras.empty()) {
    std::string err = "No cameras available";
    Logger::getLogger().log(Logger::timestamp(), Logger::Level::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  camera_ = cm_->get(cameras[0]->id());
  if (!camera_) {
    std::string err = "Failed to retrieve camera";
    Logger::getLogger().log(Logger::timestamp(), Logger::Level::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }
  if (camera_->acquire() < 0) {
    std::string err = "Failed to acquire camera";
    Logger::getLogger().log(Logger::timestamp(), Logger::Level::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  config_ = camera_->generateConfiguration({ StreamRole::VideoRecording });
  if (!config_) {
    std::string err = "Failed to generate camera configuration";
    Logger::getLogger().log(Logger::timestamp(), Logger::Level::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  StreamConfiguration &cfg = config_->at(0);
  cfg.size = { resolution.first, resolution.second };
  cfg.bufferCount = bufferCount;

  if (config_->validate() == CameraConfiguration::Invalid) {
    std::string err = "Invalid camera configuration";
    Logger::getLogger().log(Logger::timestamp(), Logger::Level::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  } else if (config_->validate() == CameraConfiguration::Adjusted) {
    Logger::getLogger().log(Logger::timestamp(), Logger::Level::WARNING, __FILE__, __LINE__, "Camera configuration adjusted to " + cfg.toString());
  }

  if (camera_->configure(config_.get()) < 0) {
    std::string err = "Failed to configure camera";
    Logger::getLogger().log(Logger::timestamp(), Logger::Level::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  allocator_ = std::make_unique<FrameBufferAllocator>(camera_);
  stream_ = config_->at(0).stream();
  if (allocator_->allocate(stream_) < 0) {
    std::string err = "Failed to allocate buffers";
    Logger::getLogger().log(Logger::timestamp(), Logger::Level::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  for (const std::unique_ptr<FrameBuffer> &buffer : allocator_->buffers(stream_)) {
    std::unique_ptr<Request> request = camera_->createRequest();
    if (!request) {
      std::string err = "Failed to create request";
      Logger::getLogger().log(Logger::timestamp(), Logger::Level::ERROR, __FILE__, __LINE__, err);
      throw std::runtime_error(err);
    }
    if (request->addBuffer(stream_, buffer.get()) < 0) {
      std::string err = "Failed to add buffer to request";
      Logger::getLogger().log(Logger::timestamp(), Logger::Level::ERROR, __FILE__, __LINE__, err);
      throw std::runtime_error(err);
    }
    requests_.push_back(std::move(request));
  }

  camera_->requestCompleted.connect(this, &CameraHandler::requestComplete);
  nextRequestIndex_.store(0);

  controls_ = std::make_unique<ControlList>();
  controls_->set(controls::FrameDurationLimits, Span<const std::int64_t, 2>({ frameDurationLimits.first, frameDurationLimits.second }));
  if (camera_->start(controls_.get()) < 0) {
    std::string err = "Failed to start camera";
    Logger::getLogger().log(Logger::timestamp(), Logger::Level::ERROR, __FILE__, __LINE__, err);
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
  size_t index = nextRequestIndex_.load();
  if (camera_->queueRequest(requests_[index].get()) < 0) {
    std::string err = "Failed to queue request";
    Logger::getLogger().log(Logger::timestamp(), Logger::Level::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }
  nextRequestIndex_.store((index + 1) % requests_.size());
}

void CameraHandler::requestComplete(Request *request) {
  if (request->status() == Request::RequestCancelled)
    return;

  Logger::getLogger().log(Logger::timestamp(), Logger::Level::INFO, __FILE__, __LINE__, "Request completed");

  const std::map<const Stream*, FrameBuffer*> &buffers = request->buffers();
  for (const auto &[stream, buffer] : buffers)
    std::cout << "Captured frame " << buffer->metadata().sequence << " from stream " << stream << std::endl;

  request->reuse(Request::ReuseBuffers);
}
