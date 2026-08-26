#include <chrono>
#include <cstdint>
#include <expected>
#include <optional>
#include <print>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "copy_gray_to_host.hpp"
#include "lens_calibration.hpp"
#include "session.hpp"

namespace {

constexpr const char* CONFIG_PATH = "cams.toml";
constexpr int BOARD_WIDTH = 9;
constexpr int BOARD_HEIGHT = 6;
constexpr float SQUARE_SIZE = 25.0f;   // mm

// calibrateCamera is expensive, so a camera that has enough frames is retried
// only every few new ones rather than after each
constexpr int RECALIBRATE_EVERY = 5;
constexpr std::chrono::microseconds POLL_INTERVAL{500};
constexpr int ESC_KEY = 27;

struct CameraProgress {
  int accepted = 0;
  int last_attempt = 0;
  bool calibrated = false;
  double error = 0.0;
};

bool all_calibrated(const std::vector<CameraProgress>& progress) {
  for (const CameraProgress& camera : progress)
    if (!camera.calibrated)
      return false;

  return true;
}

// enough frames to solve at all, and enough new ones since the last solve to
// plausibly change the answer
bool due_for_calibration(const CameraProgress& camera) {
  return !camera.calibrated
      && camera.accepted >= mocap::MIN_FRAMES
      && camera.accepted - camera.last_attempt >= RECALIBRATE_EVERY;
}

void attempt_calibration(mocap::LensCalibration& calibrator,
                         CameraProgress& camera,
                         const std::string& name) {
  camera.last_attempt = camera.accepted;
  camera.error = calibrator.calibrate();

  if (!calibrator.check_status()) {
    std::println("[{}] {} frames, reprojection error {:.4f} px, still above {:.1f}",
                 name, camera.accepted, camera.error, mocap::MIN_ERR);
    return;
  }

  camera.calibrated = true;
  std::println("[{}] calibrated: {} frames, reprojection error {:.4f} px",
               name, camera.accepted, camera.error);
}

// runs until every camera meets MIN_ERR, or the operator presses escape
// every camera gets its own window so the operator can see what each one sees
// while positioning the board. returns false if the operator aborted.
bool show(const cv::Mat& image, const std::string& window) {
  cv::imshow(window, image);
  return cv::waitKey(1) != ESC_KEY;
}

void collect_until_calibrated(std::vector<mocap::LensCalibration>& calibrators,
                              std::vector<CameraProgress>& progress,
                              const std::vector<std::string>& names) {
  cv::Mat gray;
  cv::Mat preview;
  bool running = true;

  while (running && !all_calibrated(progress)) {
    std::optional<mocap::Frameset> set = mocap::try_acquire_frameset();
    if (!set) {
      std::this_thread::sleep_for(POLL_INTERVAL);
      continue;
    }

    for (const mocap::FrameView& view : set->frames) {
      CameraProgress& camera = progress[view.camera_id];
      if (camera.calibrated)
        continue;

      if (!mocap::copy_gray_to_host(view, gray))
        continue;

      const bool found = calibrators[view.camera_id].try_frame(gray);

      cv::cvtColor(gray, preview, cv::COLOR_GRAY2BGR);
      if (found)
        calibrators[view.camera_id].draw_corners(preview);

      if (!show(preview, names[view.camera_id]))
        running = false;

      if (!found)
        continue;

      camera.accepted += 1;

      if (due_for_calibration(camera))
        attempt_calibration(calibrators[view.camera_id], camera, names[view.camera_id]);
    }

    mocap::release_frameset(*set);
  }
}

void save_all(std::vector<mocap::LensCalibration>& calibrators,
              const std::vector<CameraProgress>& progress,
              const std::vector<std::string>& names) {
  for (size_t i = 0; i < names.size(); i += 1) {
    if (!progress[i].calibrated) {
      std::println("{}: not calibrated, {} frames accepted, nothing saved",
                   names[i], progress[i].accepted);
      continue;
    }

    std::string filename = names[i] + "_calibration.yaml";
    calibrators[i].save_params(filename);
    std::println("{}: {:.4f} px from {} frames -> {}",
                 names[i], progress[i].error, progress[i].accepted, filename);
  }
}

} // namespace

int main() {
  std::expected<void, mocap::Error> started = mocap::start_session(CONFIG_PATH);
  if (!started) {
    std::println(stderr, "start_session: {}: {}",
                 started.error().detail, started.error().ec.message());
    return 1;
  }

  // the session already parsed the config, and session_config points at state
  // stop_session frees, so anything needed later is copied out now
  const mocap::Config& conf = mocap::session_config();
  const uint32_t width = conf.stream.frame_width;
  const uint32_t height = conf.stream.frame_height;

  std::vector<std::string> names;
  names.reserve(conf.cameras.size());
  for (const mocap::Camera& cam : conf.cameras)
    names.push_back(cam.name);

  std::vector<mocap::LensCalibration> calibrators;
  std::vector<CameraProgress> progress(names.size());
  calibrators.reserve(names.size());
  for (size_t i = 0; i < names.size(); i += 1)
    calibrators.emplace_back(
      static_cast<int>(width), static_cast<int>(height),
      BOARD_WIDTH, BOARD_HEIGHT, SQUARE_SIZE
    );

  std::println("hold the {}x{} board in view of each camera",
               BOARD_WIDTH, BOARD_HEIGHT);
  std::println("each needs at least {} views and under {:.1f} px error",
               mocap::MIN_FRAMES, mocap::MIN_ERR);
  std::println("escape to stop early\n");

  collect_until_calibrated(calibrators, progress, names);

  std::expected<void, mocap::Error> stopped = mocap::stop_session();
  cv::destroyAllWindows();

  std::println("");
  save_all(calibrators, progress, names);

  if (!stopped) {
    std::println(stderr, "stop_session: {}: {}",
                 stopped.error().detail, stopped.error().ec.message());
    return 1;
  }

  return 0;
}
