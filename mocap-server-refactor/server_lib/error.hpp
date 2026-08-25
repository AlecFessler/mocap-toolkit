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
Error invalid(std::string detail);

} // namespace mocap

#endif // MOCAP_ERROR_HPP
