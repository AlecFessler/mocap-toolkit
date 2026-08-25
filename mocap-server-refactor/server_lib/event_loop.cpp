#include <cstdio>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>
#include <utility>

#include "event_loop.hpp"

namespace mocap {

namespace {

constexpr size_t MAX_EVENTS = 16;

// picam refuses to send a frame larger than PKT_MAX_SIZE - 12, so a buffer
// this size can always hold one complete frame plus its header
constexpr size_t MAX_FRAME_SIZE = 262144;

// epoll hands back one 64-bit token per ready fd. We split it into what kind
// of fd fired and which camera it belongs to.
enum class source_kind : uint32_t { stop, listener, stream };

struct event_source {
  source_kind kind;
  uint32_t index;
};

uint64_t encode(source_kind kind, size_t index) {
  return (static_cast<uint64_t>(kind) << 32) | static_cast<uint32_t>(index);
}

event_source decode(uint64_t key) {
  return event_source{
    static_cast<source_kind>(key >> 32),
    static_cast<uint32_t>(key)
  };
}

} // namespace

event_loop::event_loop(unique_fd epoll_fd, unique_fd stop_fd, std::vector<stream_listener> listeners)
  : m_epoll_fd(std::move(epoll_fd)),
    m_stop_fd(std::move(stop_fd)),
    m_listeners(std::move(listeners)),
    m_events(MAX_EVENTS) {
  m_streams.reserve(m_listeners.size());
  for (size_t i = 0; i < m_listeners.size(); i++)
    m_streams.push_back(stream_state{unique_fd{}, std::vector<uint8_t>(MAX_FRAME_SIZE)});
}

std::expected<event_loop, error> event_loop::open(const std::vector<camera>& cameras) {
  unique_fd epoll_fd{epoll_create1(EPOLL_CLOEXEC)};
  if (!epoll_fd.valid())
    return std::unexpected(errno_error("failed to create epoll fd"));

  unique_fd stop_fd{eventfd(0, EFD_CLOEXEC)};
  if (!stop_fd.valid())
    return std::unexpected(errno_error("failed to create stop eventfd"));

  std::vector<stream_listener> listeners;
  listeners.reserve(cameras.size());

  for (const camera& cam : cameras) {
    std::expected<stream_listener, error> listener =
      stream_listener::open(cam.tcp_port, cam.name);
    if (!listener)
      return std::unexpected(listener.error());

    listeners.push_back(std::move(*listener));
  }

  event_loop loop(std::move(epoll_fd), std::move(stop_fd), std::move(listeners));

  std::expected<void, error> watched =
    loop.watch(loop.m_stop_fd.get(), encode(source_kind::stop, 0));
  if (!watched)
    return std::unexpected(watched.error());

  for (size_t i = 0; i < loop.m_listeners.size(); i++) {
    watched = loop.watch(loop.m_listeners[i].fd(), encode(source_kind::listener, i));
    if (!watched)
      return std::unexpected(watched.error());
  }

  return loop;
}

std::expected<void, error> event_loop::run() {
  m_stopping = false;

  while (!m_stopping) {
    std::expected<std::span<const epoll_event>, error> events = wait();
    if (!events)
      return std::unexpected(events.error());

    std::expected<void, error> serviced = service(*events);
    if (!serviced)
      return std::unexpected(serviced.error());
  }

  return {};
}

std::expected<std::span<const epoll_event>, error> event_loop::wait() {
  for (;;) {
    int ready = epoll_wait(m_epoll_fd.get(), m_events.data(), m_events.size(), -1);
    if (ready >= 0)
      return std::span<const epoll_event>(m_events.data(), static_cast<size_t>(ready));

    // a signal delivered to this thread is not a failure, just resume waiting
    if (errno != EINTR)
      return std::unexpected(errno_error("epoll_wait failed"));
  }
}

std::expected<void, error> event_loop::service(std::span<const epoll_event> events) {
  for (const epoll_event& event : events) {
    std::expected<void, error> handled = dispatch(event.data.u64);
    if (!handled)
      return std::unexpected(handled.error());
  }

  return {};
}

std::expected<void, error> event_loop::dispatch(uint64_t key) {
  const event_source source = decode(key);

  switch (source.kind) {
    case source_kind::stop:
      m_stopping = true;
      return {};

    case source_kind::listener:
      return accept_camera(source.index);

    case source_kind::stream:
      return read_camera(source.index);
  }

  return std::unexpected(invalid("unknown epoll event source"));
}

std::expected<void, error> event_loop::watch(int fd, uint64_t key) const {
  epoll_event ev{};
  ev.events = EPOLLIN;
  ev.data.u64 = key;

  if (epoll_ctl(m_epoll_fd.get(), EPOLL_CTL_ADD, fd, &ev) < 0)
    return std::unexpected(errno_error("failed to add fd to epoll set"));

  return {};
}

std::expected<void, error> event_loop::accept_camera(size_t index) {
  std::expected<unique_fd, error> conn = m_listeners[index].accept();
  if (!conn)
    return std::unexpected(conn.error());

  std::expected<void, error> watched = watch(conn->get(), encode(source_kind::stream, index));
  if (!watched)
    return std::unexpected(watched.error());

  std::printf("[%s] connected\n", m_listeners[index].name().c_str());
  m_streams[index].fd = std::move(*conn);
  return {};
}

std::expected<void, error> event_loop::read_camera(size_t index) {
  stream_state& stream = m_streams[index];

  ssize_t got = read(stream.fd.get(), stream.buffer.data(), stream.buffer.size());
  if (got < 0)
    return std::unexpected(errno_error("failed to read from stream"));

  if (got == 0) {
    std::printf("[%s] disconnected\n", m_listeners[index].name().c_str());
    epoll_ctl(m_epoll_fd.get(), EPOLL_CTL_DEL, stream.fd.get(), nullptr);
    stream.fd.reset();
    return {};
  }

  return {};
}

std::expected<void, error> event_loop::stop() const {
  uint64_t one = 1;
  if (write(m_stop_fd.get(), &one, sizeof(one)) != sizeof(one))
    return std::unexpected(errno_error("failed to signal stop eventfd"));

  return {};
}

} // namespace mocap
