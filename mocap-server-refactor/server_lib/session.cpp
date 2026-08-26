#include <chrono>
#include <cstdio>
#include <cstdlib>
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
// TCP stream up, so the start time is placed far enough ahead for every Camera
// to be armed before it arrives
constexpr std::chrono::seconds START_DELAY{2};

struct SessionState {
  Config conf;
  ControlSocket control;
  EventLoop loop;
  std::thread thread;
};

std::optional<SessionState> g_session;

// calling into a session that was never started is a bug in the caller, not a
// runtime condition, so there is nothing to hand back that they could act on.
// exiting rather than aborting keeps the diagnostic, since abort does not
// flush stdio.
[[noreturn]] void no_session(const char* function) {
  std::fprintf(stderr, "[session] %s called with no session running\n", function);
  std::exit(EXIT_FAILURE);
}

uint64_t start_timestamp() {
  std::chrono::nanoseconds now =
    std::chrono::system_clock::now().time_since_epoch();
  return static_cast<uint64_t>((now + START_DELAY).count());
}

} // namespace

std::expected<void, Error> start_session(const std::filesystem::path& config_path) {
  if (g_session)
    return std::unexpected(invalid("session already started"));

  std::expected<Config, Error> conf = parse_config(config_path);
  if (!conf)
    return std::unexpected(conf.error());

  std::expected<ControlSocket, Error> control =
    ControlSocket::open(conf->control.broadcast, conf->control.port);
  if (!control)
    return std::unexpected(control.error());

  // computed before the loop opens so the frameset builder can reject frames
  // captured before this session began
  const uint64_t timestamp = start_timestamp();

  // listeners must be bound before the cameras are told to start, otherwise
  // their first connect is refused and they back off for a retry interval
  std::expected<EventLoop, Error> loop = EventLoop::open(*conf, timestamp);
  if (!loop)
    return std::unexpected(loop.error());

  std::expected<void, Error> started = control->broadcast_start(timestamp);
  if (!started)
    return std::unexpected(started.error());

  g_session.emplace(std::move(*conf), std::move(*control), std::move(*loop), std::thread{});
  g_session->thread = std::thread([] { (void)g_session->loop.run(); });

  return {};
}

std::expected<void, Error> stop_session() {
  // stopping a session that was never started is not something a caller can
  // act on, so it is a no op rather than an error they have to handle
  if (!g_session) {
    std::fprintf(stderr, "[session] stop_session with no session running\n");
    return {};
  }

  std::expected<void, Error> stopped = g_session->control.broadcast_stop();

  // the eventfd is the only way to wake the loop, so if the write fails the
  // join below would block forever. there is no path back from that, and
  // checking after the join would never run.
  std::expected<void, Error> signalled = g_session->loop.stop();
  if (!signalled) {
    // not exit: the loop thread is still running and joinable, so running its
    // destructor on the way out would terminate anyway
    std::fprintf(stderr, "[session] cannot signal the event loop to stop: %s: %s\n",
                 signalled.error().detail.c_str(),
                 signalled.error().ec.message().c_str());
    std::abort();
  }

  if (g_session->thread.joinable())
    g_session->thread.join();

  g_session.reset();

  // the only failure a caller can act on: the cameras were never told to stop,
  // so they keep capturing until their stream write fails
  if (!stopped)
    return std::unexpected(stopped.error());

  return {};
}

const Config& session_config() {
  if (!g_session)
    no_session("session_config");

  return g_session->conf;
}

std::optional<Frameset> try_acquire_frameset() {
  if (!g_session)
    no_session("try_acquire_frameset");

  return g_session->loop.pool().try_acquire();
}

void release_frameset(const Frameset& set) {
  if (!g_session)
    no_session("release_frameset");

  g_session->loop.pool().release(set.slot);
}

} // namespace mocap
