#ifndef MOCAP_ERROR_HPP
#define MOCAP_ERROR_HPP

#include <string>
#include <system_error>

namespace mocap {

struct error {
  std::error_code ec;
  std::string detail;
};

error errno_error(std::string detail);
error invalid(std::string detail);

} // namespace mocap

#endif // MOCAP_ERROR_HPP
