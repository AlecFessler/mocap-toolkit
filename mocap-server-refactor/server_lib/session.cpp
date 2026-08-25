#include <chrono>
#include <cstdint>
#include <optional>
#include <thread>
#include <utility>

#include "config.hpp"
#include "control.hpp"
#include "event_loop.hpp"
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
  event_loop loop;
  std::thread thread;
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

  // listeners must be bound before the cameras are told to start, otherwise
  // their first connect is refused and they back off for a retry interval
  std::expected<event_loop, error> loop = event_loop::open(conf->cameras);
  if (!loop)
    return std::unexpected(loop.error());

  std::expected<void, error> started = control->broadcast_start(start_timestamp());
  if (!started)
    return std::unexpected(started.error());

  g_session.emplace(std::move(*conf), std::move(*control), std::move(*loop), std::thread{});
  g_session->thread = std::thread([] { (void)g_session->loop.run(); });

  return {};
}

std::expected<void, error> session_stop() {
  if (!g_session)
    return std::unexpected(invalid("session not started"));

  std::expected<void, error> stopped = g_session->control.broadcast_stop();

  std::expected<void, error> signalled = g_session->loop.stop();
  if (g_session->thread.joinable())
    g_session->thread.join();

  g_session.reset();

  if (!stopped)
    return std::unexpected(stopped.error());
  if (!signalled)
    return std::unexpected(signalled.error());

  return {};
}

} // namespace mocap
