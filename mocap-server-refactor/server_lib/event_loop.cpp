#include <cstdio>
#include <cstring>
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
constexpr size_t FRAME_HEADER_SIZE = 12;
constexpr size_t MAX_PAYLOAD_SIZE = MAX_FRAME_SIZE - FRAME_HEADER_SIZE;

// NVDEC surface pool per stream. Every frame handed out holds one, so this is
// the ceiling on frames in flight before the decoder stalls.
constexpr uint32_t DECODE_SURFACES = 16;

// picam frames a packet as [uint64 capture timestamp][uint32 payload size]
struct FrameHeader {
  uint64_t timestamp;
  uint32_t size;
};

FrameHeader read_frame_header(const uint8_t* data) {
  FrameHeader header;
  std::memcpy(&header.timestamp, data, sizeof(header.timestamp));
  std::memcpy(&header.size, data + sizeof(header.timestamp), sizeof(header.size));
  return header;
}

// epoll hands back one 64-bit token per ready fd. We split it into what kind
// of fd fired and which Camera it belongs to.
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

EventLoop::EventLoop(UniqueFd epoll_fd, UniqueFd stop_fd, HwContext hw,
                     std::vector<StreamListener> listeners, std::vector<StreamState> streams)
  : m_epoll_fd(std::move(epoll_fd)),
    m_stop_fd(std::move(stop_fd)),
    m_hw(std::move(hw)),
    m_listeners(std::move(listeners)),
    m_streams(std::move(streams)),
    m_events(MAX_EVENTS) {}

std::expected<EventLoop, Error> EventLoop::open(const Config& conf) {
  UniqueFd epoll_fd{epoll_create1(EPOLL_CLOEXEC)};
  if (!epoll_fd.valid())
    return std::unexpected(errno_error("failed to create epoll fd"));

  UniqueFd stop_fd{eventfd(0, EFD_CLOEXEC)};
  if (!stop_fd.valid())
    return std::unexpected(errno_error("failed to create stop eventfd"));

  std::expected<HwContext, Error> hw = HwContext::open();
  if (!hw)
    return std::unexpected(hw.error());

  std::vector<StreamListener> listeners;
  std::vector<StreamState> streams;
  listeners.reserve(conf.cameras.size());
  streams.reserve(conf.cameras.size());

  for (const Camera& cam : conf.cameras) {
    std::expected<StreamListener, Error> listener =
      StreamListener::open(cam.tcp_port, cam.name);
    if (!listener)
      return std::unexpected(listener.error());

    std::expected<Decoder, Error> decoder =
      Decoder::open(conf.stream, DECODE_SURFACES, *hw);
    if (!decoder)
      return std::unexpected(decoder.error());

    listeners.push_back(std::move(*listener));
    streams.push_back(StreamState{
      UniqueFd{}, std::vector<uint8_t>(MAX_PAYLOAD_SIZE), {}, 0, 0, 0, std::move(*decoder)
    });
  }

  EventLoop loop(std::move(epoll_fd), std::move(stop_fd), std::move(*hw),
                 std::move(listeners), std::move(streams));

  std::expected<void, Error> watched =
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

std::expected<void, Error> EventLoop::run() {
  m_stopping = false;

  while (!m_stopping) {
    std::expected<std::span<const epoll_event>, Error> events = wait();
    if (!events)
      return std::unexpected(events.error());

    std::expected<void, Error> serviced = service(*events);
    if (!serviced)
      return std::unexpected(serviced.error());
  }

  return {};
}

std::expected<std::span<const epoll_event>, Error> EventLoop::wait() {
  for (;;) {
    int ready = epoll_wait(m_epoll_fd.get(), m_events.data(), m_events.size(), -1);
    if (ready >= 0)
      return std::span<const epoll_event>(m_events.data(), static_cast<size_t>(ready));

    // a signal delivered to this thread is not a failure, just resume waiting
    if (errno != EINTR)
      return std::unexpected(errno_error("epoll_wait failed"));
  }
}

std::expected<void, Error> EventLoop::service(std::span<const epoll_event> events) {
  for (const epoll_event& event : events) {
    std::expected<void, Error> handled = dispatch(event.data.u64);
    if (!handled)
      return std::unexpected(handled.error());
  }

  return {};
}

std::expected<void, Error> EventLoop::dispatch(uint64_t key) {
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

std::expected<void, Error> EventLoop::watch(int fd, uint64_t key) const {
  epoll_event ev{};
  ev.events = EPOLLIN;
  ev.data.u64 = key;

  if (epoll_ctl(m_epoll_fd.get(), EPOLL_CTL_ADD, fd, &ev) < 0)
    return std::unexpected(errno_error("failed to add fd to epoll set"));

  return {};
}

std::expected<void, Error> EventLoop::accept_camera(size_t index) {
  std::expected<UniqueFd, Error> conn = m_listeners[index].accept();
  if (!conn)
    return std::unexpected(conn.error());

  std::expected<void, Error> watched = watch(conn->get(), encode(source_kind::stream, index));
  if (!watched)
    return std::unexpected(watched.error());

  std::printf("[%s] connected\n", m_listeners[index].name().c_str());
  m_streams[index].fd = std::move(*conn);
  return {};
}

std::expected<void, Error> EventLoop::read_camera(size_t index) {
  if (m_streams[index].expected == 0)
    return read_header(index);

  return read_payload(index);
}

// reads at most the bytes still missing, so a read never crosses into the
// next frame. whatever else arrived stays in the socket buffer and wakes us
// again, which keeps every read aligned to a frame boundary.
std::expected<size_t, Error> EventLoop::fill(size_t index, uint8_t* dst, size_t want) {
  StreamState& stream = m_streams[index];

  ssize_t got = read(stream.fd.get(), dst + stream.filled, want - stream.filled);
  if (got < 0)
    return std::unexpected(errno_error("failed to read from stream"));

  if (got == 0) {
    std::printf("[%s] disconnected\n", m_listeners[index].name().c_str());
    epoll_ctl(m_epoll_fd.get(), EPOLL_CTL_DEL, stream.fd.get(), nullptr);
    stream.fd.reset();
    stream.filled = 0;
    stream.expected = 0;
    return 0;
  }

  stream.filled += static_cast<size_t>(got);
  return stream.filled;
}

std::expected<void, Error> EventLoop::read_header(size_t index) {
  StreamState& stream = m_streams[index];

  std::expected<size_t, Error> filled = fill(index, stream.header, FRAME_HEADER_SIZE);
  if (!filled)
    return std::unexpected(filled.error());

  if (*filled < FRAME_HEADER_SIZE)
    return {};

  const FrameHeader header = read_frame_header(stream.header);
  if (header.size == 0 || header.size > MAX_PAYLOAD_SIZE)
    return std::unexpected(invalid("frame size outside protocol bounds"));

  stream.timestamp = header.timestamp;
  stream.expected = header.size;
  stream.filled = 0;
  return {};
}

std::expected<void, Error> EventLoop::read_payload(size_t index) {
  StreamState& stream = m_streams[index];

  std::expected<size_t, Error> filled = fill(index, stream.buffer.data(), stream.expected);
  if (!filled)
    return std::unexpected(filled.error());

  if (*filled < stream.expected)
    return {};

  return submit_frame(index);
}

std::expected<void, Error> EventLoop::submit_frame(size_t index) {
  StreamState& stream = m_streams[index];

  std::expected<void, Error> sent = stream.decoder.send_packet(
    std::span<const uint8_t>(stream.buffer.data(), stream.expected),
    stream.timestamp
  );
  if (!sent)
    return std::unexpected(sent.error());

  stream.filled = 0;
  stream.expected = 0;

  // the decoder stops accepting input once its queue fills, so every send has
  // to be followed by draining whatever became ready
  return drain_frames(index);
}

std::expected<void, Error> EventLoop::drain_frames(size_t index) {
  StreamState& stream = m_streams[index];

  for (;;) {
    std::expected<std::optional<DecodedFrame>, Error> frame = stream.decoder.receive_frame();
    if (!frame)
      return std::unexpected(frame.error());

    if (!frame->has_value())
      return {};

    // nothing consumes frames yet, so the frame is dropped here and its
    // surface goes straight back into the decoder pool

  }
}

std::expected<void, Error> EventLoop::stop() const {
  uint64_t one = 1;
  if (write(m_stop_fd.get(), &one, sizeof(one)) != sizeof(one))
    return std::unexpected(errno_error("failed to signal stop eventfd"));

  return {};
}

} // namespace mocap
