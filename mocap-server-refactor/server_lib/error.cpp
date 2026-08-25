#include <cerrno>
#include <utility>

#include "error.hpp"

namespace mocap {

Error errno_error(std::string detail) {
  return Error{
    std::error_code(errno ? errno : EIO, std::generic_category()),
    std::move(detail)
  };
}

Error invalid(std::string detail) {
  return Error{
    std::make_error_code(std::errc::invalid_argument),
    std::move(detail)
  };
}

} // namespace mocap
