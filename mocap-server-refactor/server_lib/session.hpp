#ifndef MOCAP_SESSION_HPP
#define MOCAP_SESSION_HPP

#include <expected>
#include <filesystem>

#include "error.hpp"

namespace mocap {

// Starts a capture session: reads the camera config, then broadcasts a start
// timestamp to every camera at once. Cameras arm their capture timers against
// that absolute time, so they begin in lockstep regardless of delivery jitter.
//
// One session per process. Not thread safe.
std::expected<void, error> session_start(const std::filesystem::path& config_path);

// Broadcasts the stop sentinel and tears down the session. Safe to call only
// on a started session.
std::expected<void, error> session_stop();

} // namespace mocap

#endif // MOCAP_SESSION_HPP
