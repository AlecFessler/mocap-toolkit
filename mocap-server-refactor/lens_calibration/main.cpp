#include <chrono>
#include <cstdint>
#include <expected>
#include <print>
#include <thread>

#include "session.hpp"

constexpr const char* CONFIG_PATH = "cams.toml";
constexpr std::chrono::seconds CAPTURE_WINDOW{5};
constexpr std::chrono::microseconds POLL_INTERVAL{500};

int main() {
  std::expected<void, mocap::Error> started = mocap::session_start(CONFIG_PATH);
  if (!started) {
    std::println(stderr, "session_start: {}: {}",
                 started.error().detail, started.error().ec.message());
    return 1;
  }
  std::println("session started, capturing for {}s", CAPTURE_WINDOW.count());

  std::chrono::steady_clock::time_point deadline =
    std::chrono::steady_clock::now() + CAPTURE_WINDOW;

  uint64_t sets = 0;
  uint64_t partial = 0;
  uint64_t first_ts = 0;
  uint64_t last_ts = 0;

  while (std::chrono::steady_clock::now() < deadline) {
    std::optional<mocap::Frameset> set = mocap::try_acquire_frameset();
    if (!set) {
      std::this_thread::sleep_for(POLL_INTERVAL);
      continue;
    }

    if (sets == 0) {
      first_ts = set->timestamp;
      std::println("first set: ts={} frames={}", set->timestamp, set->frames.size());
      for (const mocap::FrameView& f : set->frames)
        std::println("  cam{} dev_ptr={} pitch={}",
                     f.camera_id, static_cast<const void*>(f.device_ptr), f.pitch);
    }

    if (set->frames.size() < 3)
      partial++;

    last_ts = set->timestamp;
    sets++;

    mocap::release_frameset(*set);
  }

  std::expected<void, mocap::Error> stopped = mocap::session_stop();
  if (!stopped) {
    std::println(stderr, "session_stop: {}: {}",
                 stopped.error().detail, stopped.error().ec.message());
    return 1;
  }

  std::println("\nframesets: {} ({} partial)", sets, partial);
  if (sets > 1)
    std::println("span: {:.3f} s, mean interval {:.3f} ms",
                 (last_ts - first_ts) / 1e9,
                 (last_ts - first_ts) / 1e6 / static_cast<double>(sets - 1));

  return 0;
}
