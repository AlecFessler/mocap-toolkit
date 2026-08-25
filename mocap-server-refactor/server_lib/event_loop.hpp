#ifndef MOCAP_EVENT_LOOP_HPP
#define MOCAP_EVENT_LOOP_HPP

#include <cstdint>
#include <expected>
#include <span>
#include <sys/epoll.h>
#include <vector>

#include "config.hpp"
#include "error.hpp"
#include "fd.hpp"
#include "stream.hpp"

namespace mocap {

// Per-camera receive state. Streams interleave, so a partial frame from one
// camera has to survive reads from the others -- the buffer cannot be shared.
struct stream_state {
  unique_fd fd;
  std::vector<uint8_t> buffer;
};

// Single-threaded epoll loop owning every camera stream. One thread services
// all cameras, so the work done per wakeup needs to stay bounded.
class event_loop {
public:
  static std::expected<event_loop, error> open(const std::vector<camera>& cameras);

  event_loop(event_loop&&) noexcept = default;
  event_loop& operator=(event_loop&&) noexcept = default;

  // blocks until stop() is called
  std::expected<void, error> run();

  // wakes run() from another thread
  std::expected<void, error> stop() const;

private:
  event_loop(unique_fd epoll_fd, unique_fd stop_fd, std::vector<stream_listener> listeners);

  std::expected<void, error> watch(int fd, uint64_t key) const;

  std::expected<std::span<const epoll_event>, error> wait();
  std::expected<void, error> service(std::span<const epoll_event> events);
  std::expected<void, error> dispatch(uint64_t key);

  std::expected<void, error> accept_camera(size_t index);
  std::expected<void, error> read_camera(size_t index);

  unique_fd m_epoll_fd;
  unique_fd m_stop_fd;
  std::vector<stream_listener> m_listeners;
  std::vector<stream_state> m_streams;
  std::vector<epoll_event> m_events;
  bool m_stopping = false;
};

} // namespace mocap

#endif // MOCAP_EVENT_LOOP_HPP
