#include <unistd.h>
#include <utility>

#include "fd.hpp"

namespace mocap {

unique_fd::unique_fd(int fd) : m_fd(fd) {}

unique_fd::~unique_fd() {
  reset();
}

unique_fd::unique_fd(unique_fd&& other) noexcept
  : m_fd(std::exchange(other.m_fd, -1)) {}

unique_fd& unique_fd::operator=(unique_fd&& other) noexcept {
  if (this != &other) {
    reset();
    m_fd = std::exchange(other.m_fd, -1);
  }
  return *this;
}

void unique_fd::reset() {
  if (m_fd >= 0) {
    close(m_fd);
    m_fd = -1;
  }
}

} // namespace mocap
