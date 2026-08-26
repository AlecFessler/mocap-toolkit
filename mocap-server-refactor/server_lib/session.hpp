#ifndef MOCAP_SESSION_HPP
#define MOCAP_SESSION_HPP

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>

#include "config.hpp"
#include "error.hpp"
#include "visibility.hpp"

namespace mocap {

// One camera's decoded frame, still resident on the GPU. NV12, chroma follows
// the luma plane at pitch * height.
struct FrameView {
  uint8_t camera_id;
  const uint8_t* device_ptr;
  uint32_t width;
  uint32_t height;
  uint32_t pitch;    // padded wider than width for alignment, so copies stride
};

// A set of frames that share a capture timestamp. May hold fewer than one per
// camera if a camera missed the window, so read frames.size() rather than
// assuming a full set. Every acquired set must be handed back to
// release_frameset or its GPU surfaces are never reused.
struct Frameset {
  uint64_t timestamp;
  std::span<const FrameView> frames;
  uint32_t slot;   // opaque, hand back to release_frameset
};

// Starts a capture session: reads the camera config, then broadcasts a start
// timestamp to every camera at once. Cameras arm their capture timers against
// that absolute time, so they begin in lockstep regardless of delivery jitter.
//
// One session per process. Not thread safe.
MOCAP_API std::expected<void, Error> start_session(const std::filesystem::path& config_path);

// Broadcasts the stop sentinel and tears down the session. Safe to call only
// on a started session.
MOCAP_API std::expected<void, Error> stop_session();

// Takes the oldest ready frameset, or nullopt if none is ready. Safe to call
// from a different thread than the one that started the session.
// The config the running session was started from. Owned by the library and
// valid until stop_session, so copy out anything needed after that.
//
// Calling this without a running session is a caller bug rather than a
// condition to handle, so it aborts rather than handing back a null nobody
// checks.
MOCAP_API const Config& session_config();

MOCAP_API std::optional<Frameset> try_acquire_frameset();

// Returns a frameset's GPU surfaces to the decoders. Hold sets briefly: the
// surface pool is fixed, and the library drops the oldest ready set when it
// runs out of slots.
MOCAP_API void release_frameset(const Frameset& set);

} // namespace mocap

#endif // MOCAP_SESSION_HPP
