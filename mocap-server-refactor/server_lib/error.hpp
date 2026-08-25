#ifndef MOCAP_ERROR_HPP
#define MOCAP_ERROR_HPP

#include <string>
#include <system_error>

namespace mocap {

struct Error {
  std::error_code ec;
  std::string detail;
};

Error errno_error(std::string detail);

// not a failure: the operation made no progress and should be retried when
// its fd next wakes up. short circuits an and_then chain without ending it.
Error retry();
bool is_retry(const Error& err);
Error invalid(std::string detail);

} // namespace mocap

#endif // MOCAP_ERROR_HPP
