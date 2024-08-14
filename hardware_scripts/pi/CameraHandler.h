#ifndef CAMERAHANDLER_H
#define CAMERAHANDLER_H

#include <atomic>
#include <memory>
#include <vector>

#include <libcamera/libcamera.h>

using namespace libcamera;

class CameraHandler {
public:
  CameraHandler(
    std::pair<unsigned int, unsigned int> resolution,
    int bufferCount,
    std::pair<std::int64_t, std::int64_t> frameDurationLimits
  );
  ~CameraHandler();
  CameraHandler(const CameraHandler&) = delete;
  CameraHandler& operator=(const CameraHandler&) = delete;
  CameraHandler(CameraHandler&&) = delete;
  CameraHandler& operator=(CameraHandler&&) = delete;
  void queueRequest();
private:
  std::unique_ptr<CameraManager> cm_;
  std::shared_ptr<Camera> camera_;
  std::unique_ptr<CameraConfiguration> config_;
  std::unique_ptr<FrameBufferAllocator> allocator_;
  std::unique_ptr<ControlList> controls_;
  Stream *stream_;
  std::vector<std::unique_ptr<Request>> requests_;
  std::atomic<size_t> nextRequestIndex_;
  void requestComplete(Request *request);
};

#endif // CAMERAHANDLER_H
