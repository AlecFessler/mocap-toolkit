#include <chrono>
#include <expected>
#include <print>
#include <thread>

#include "session.hpp"

constexpr const char* CONFIG_PATH = "cams.toml";
constexpr std::chrono::seconds CAPTURE_WINDOW{5};

int main() {
  std::expected<void, mocap::error> started = mocap::session_start(CONFIG_PATH);
  if (!started) {
    std::println(stderr, "session_start: {}: {}",
                 started.error().detail, started.error().ec.message());
    return 1;
  }
  std::println("session started, capturing for {}s", CAPTURE_WINDOW.count());

  std::this_thread::sleep_for(CAPTURE_WINDOW);

  std::expected<void, mocap::error> stopped = mocap::session_stop();
  if (!stopped) {
    std::println(stderr, "session_stop: {}: {}",
                 stopped.error().detail, stopped.error().ec.message());
    return 1;
  }
  std::println("session stopped");

  return 0;
}
