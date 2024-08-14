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
  if (cm_->start() < 0)
    throw std::runtime_error("Failed to start camera manager");
  auto cameras = cm_->cameras();
  if (cameras.empty())
    throw std::runtime_error("No cameras available");
  camera_ = cm_->get(cameras[0]->id());
  if (!camera_)
    throw std::runtime_error("Failed to retrieve camera");
  if (camera_->acquire() < 0)
    throw std::runtime_error("Failed to acquire camera");

  config_ = camera_->generateConfiguration({ StreamRole::VideoRecording });
  if (!config_)
    throw std::runtime_error("Failed to generate camera configuration");

  config_->at(0).size = { resolution.first, resolution.second };
  config_->at(0).bufferCount = bufferCount;

  if (config_->validate() == CameraConfiguration::Invalid)
    throw std::runtime_error("Invalid camera configuration");
  if (camera_->configure(config_.get()) < 0)
    throw std::runtime_error("Failed to configure camera");

  allocator_ = std::make_unique<FrameBufferAllocator>(camera_);
  stream_ = config_->at(0).stream();
  if (allocator_->allocate(stream_) < 0)
    throw std::runtime_error("Failed to allocate buffers");
  for (const std::unique_ptr<FrameBuffer> &buffer : allocator_->buffers(stream_)) {
    std::unique_ptr<Request> request = camera_->createRequest();
    if (!request)
      throw std::runtime_error("Failed to create request");
    if (request->addBuffer(stream_, buffer.get()) < 0)
      throw std::runtime_error("Failed to add buffer to request");
    requests_.push_back(std::move(request));
  }

  camera_->requestCompleted.connect(this, &CameraHandler::requestComplete);
  nextRequestIndex_.store(0);

  controls_ = std::make_unique<ControlList>();
  controls_->set(controls::FrameDurationLimits, Span<const std::int64_t, 2>({ frameDurationLimits.first, frameDurationLimits.second }));
  if (camera_->start(controls_.get()) < 0)
    throw std::runtime_error("Failed to start camera");
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
  if (camera_->queueRequest(requests_[index].get()) < 0)
    throw std::runtime_error("Failed to queue request");
  nextRequestIndex_.store((index + 1) % requests_.size());
}

void CameraHandler::requestComplete(Request *request) {
  if (request->status() == Request::RequestCancelled)
    return;

  Logger::getLogger().log(LOG_EVENT(Logger::timestamp(), std::string("request complete")), Logger::Level::INFO);

  const std::map<const Stream*, FrameBuffer*> &buffers = request->buffers();
  for (const auto &[stream, buffer] : buffers)
    std::cout << "Captured frame " << buffer->metadata().sequence << " from stream " << stream << std::endl;

  request->reuse(Request::ReuseBuffers);
}
