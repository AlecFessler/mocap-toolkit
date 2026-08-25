#include <cerrno>
#include <utility>

#include "error.hpp"

namespace mocap {

error errno_error(std::string detail) {
  return error{
    std::error_code(errno ? errno : EIO, std::generic_category()),
    std::move(detail)
  };
}

error invalid(std::string detail) {
  return error{
    std::make_error_code(std::errc::invalid_argument),
    std::move(detail)
  };
}

} // namespace mocap
