#include <chrono>
#include <cstdint>
#include <optional>

#include "config.hpp"
#include "control.hpp"
#include "session.hpp"

namespace mocap {
namespace {

// cameras need roughly a second between receiving the timestamp and having a
// TCP stream up, so the start time is placed far enough ahead for every camera
// to be armed before it arrives
constexpr std::chrono::seconds START_DELAY{2};

struct session_state {
  config conf;
  control_socket control;
};

std::optional<session_state> g_session;

uint64_t start_timestamp() {
  std::chrono::nanoseconds now =
    std::chrono::system_clock::now().time_since_epoch();
  return static_cast<uint64_t>((now + START_DELAY).count());
}

} // namespace

std::expected<void, error> session_start(const std::filesystem::path& config_path) {
  if (g_session)
    return std::unexpected(invalid("session already started"));

  std::expected<config, error> conf = parse_config(config_path);
  if (!conf)
    return std::unexpected(conf.error());

  std::expected<control_socket, error> control =
    control_socket::open(conf->control_broadcast, conf->control_port);
  if (!control)
    return std::unexpected(control.error());

  std::expected<void, error> started = control->broadcast_start(start_timestamp());
  if (!started)
    return std::unexpected(started.error());

  g_session.emplace(std::move(*conf), std::move(*control));
  return {};
}

std::expected<void, error> session_stop() {
  if (!g_session)
    return std::unexpected(invalid("session not started"));

  std::expected<void, error> stopped = g_session->control.broadcast_stop();
  g_session.reset();
  if (!stopped)
    return std::unexpected(stopped.error());

  return {};
}

} // namespace mocap
