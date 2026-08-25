#ifndef MOCAP_EVENT_LOOP_HPP
#define MOCAP_EVENT_LOOP_HPP

#include <cstdint>
#include <expected>
#include <span>
#include <sys/epoll.h>
#include <vector>

#include "config.hpp"
#include "decoder.hpp"
#include "error.hpp"
#include "fd.hpp"
#include "stream.hpp"

namespace mocap {

// Per-Camera receive state. Streams interleave, so a partial frame from one
// Camera has to survive reads from the others, the buffer cannot be shared.
struct StreamState {
  UniqueFd fd;
  std::vector<uint8_t> buffer;
  uint8_t header[12];

  // bytes of the header or payload gathered so far. reads are bounded to
  // exactly what is still missing, so a read never crosses a frame boundary
  size_t filled = 0;
  uint32_t expected = 0;   // payload size, 0 while still gathering the header
  uint64_t timestamp = 0;

  Decoder decoder;
};

// Single-threaded epoll loop owning every Camera stream. One thread services
// all cameras, so the work done per wakeup needs to stay bounded.
class EventLoop {
public:
  static std::expected<EventLoop, Error> open(const Config& conf);

  EventLoop(EventLoop&&) noexcept = default;
  EventLoop& operator=(EventLoop&&) noexcept = default;

  // blocks until stop() is called
  std::expected<void, Error> run();

  // wakes run() from another thread
  std::expected<void, Error> stop() const;

private:
  EventLoop(UniqueFd epoll_fd, UniqueFd stop_fd, HwContext hw,
            std::vector<StreamListener> listeners, std::vector<StreamState> streams);

  std::expected<void, Error> watch(int fd, uint64_t key) const;

  std::expected<std::span<const epoll_event>, Error> wait();
  std::expected<void, Error> service(std::span<const epoll_event> events);
  std::expected<void, Error> dispatch(uint64_t key);

  std::expected<void, Error> accept_camera(size_t index);
  std::expected<void, Error> read_camera(size_t index);
  std::expected<void, Error> read_header(size_t index);
  std::expected<void, Error> read_payload(size_t index);
  std::expected<void, Error> submit_frame(size_t index);
  std::expected<void, Error> drain_frames(size_t index);
  std::expected<size_t, Error> fill(size_t index, uint8_t* dst, size_t want);

  UniqueFd m_epoll_fd;
  UniqueFd m_stop_fd;
  HwContext m_hw;
  std::vector<StreamListener> m_listeners;
  std::vector<StreamState> m_streams;
  std::vector<epoll_event> m_events;
  bool m_stopping = false;
};

} // namespace mocap

#endif // MOCAP_EVENT_LOOP_HPP
