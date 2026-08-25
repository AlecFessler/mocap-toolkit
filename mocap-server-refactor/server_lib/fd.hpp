#ifndef MOCAP_FD_HPP
#define MOCAP_FD_HPP

namespace mocap {

// Owns a file descriptor. Move-only: copying would give two owners one
// descriptor, and the second close would hit whatever fd number the kernel
// had since recycled.
class unique_fd {
public:
  unique_fd() = default;
  explicit unique_fd(int fd);
  ~unique_fd();

  unique_fd(unique_fd&& other) noexcept;
  unique_fd& operator=(unique_fd&& other) noexcept;
  unique_fd(const unique_fd&) = delete;
  unique_fd& operator=(const unique_fd&) = delete;

  int get() const { return m_fd; }
  bool valid() const { return m_fd >= 0; }
  void reset();

private:
  int m_fd = -1;
};

} // namespace mocap

#endif // MOCAP_FD_HPP
