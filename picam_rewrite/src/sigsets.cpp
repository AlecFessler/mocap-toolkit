// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <csignal>
#include <cerrno>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>

#include "logging.hpp"
#include "sigsets.hpp"

sigset_t setup_sigwait(int signum) {
  sigset_t sigset;

  sigemptyset(&sigset);
  sigaddset(&sigset, signum);
  int status = sigprocmask(SIG_BLOCK, &sigset, nullptr);
  if (status == -1) {
    std::string err_msg =
      "Failed to set sigmask: "
      + std::string(strerror(errno));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  return sigset;
}

sigset_t setup_sigwait(std::initializer_list<int> signums) {
  sigset_t sigset;
  sigemptyset(&sigset);

  for (int signum : signums)
    sigaddset(&sigset, signum);

  int status = sigprocmask(SIG_BLOCK, &sigset, nullptr);
  if (status == -1) {
    std::string err_msg =
      "Failed to set sigmask: "
      + std::string(strerror(errno));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  return sigset;
}

void setup_sig_handler(int signum, signal_handler_t handler) {
  struct sigaction sa{};
  sa.sa_handler = handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESTART;
  int status = sigaction(signum, &sa, nullptr);
  if (status == -1) {
    std::string err_msg =
      "Failed to set signal handler: "
      + std::string(strerror(errno));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }
}
