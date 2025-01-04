// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <string>
#include <stdexcept>
#include <sys/types.h>
#include <time.h>

#include "interval_timer.hpp"
#include "logging.hpp"

IntervalTimer::IntervalTimer(
  std::chrono::nanoseconds initial_timestamp,
  std::chrono::nanoseconds interval,
  int signum
) :
  initial_timestamp(initial_timestamp),
  interval(interval),
  counter(0) {
  /**
   * Initializes a posix timer that emits a specified signal
   * on a specified interval. The timer requires the caller
   * to meet the soft real time requirements within the interval
   * if they wish to respond to every interval along the capture
   * schedule. The arm timer function will adjust if the next
   * timestamp has already passed. The caller is responsible
   * for setting up a mechanism to respond to the signal.
   *
   * The timer is network protocol sync aware across a network,
   * which is to say, if you have multiple devices on a network
   * with real time clocks synced by NTP or PTP, this timer will
   * sync up the signal emission across the devices with exceptional
   * precision so long as all devices received the same initial timestamp
   * and the same interval.
   */
  struct sigevent sev;
  sev.sigev_notify = SIGEV_SIGNAL;
  sev.sigev_signo = signum;

  int status = timer_create(
    CLOCK_MONOTONIC,
    &sev,
    &timerid
  );
  if (status == -1) {
    std::string err_msg =
      "Failed to create timer: "
      + std::string(strerror(errno));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }
}

IntervalTimer::~IntervalTimer() {
  timer_delete(timerid);
}

std::chrono::nanoseconds IntervalTimer::arm_timer() {
  /**
   * Arms the timer for the next interval, or adjusts
   * it if the next interval has already passed. This
   * function results in a capture interval determined
   * by a simple formula:
   *
   *   Tn = T0 + interval * counter
   *
   * When multiple devices across a network are synced
   * with PTP or NTP, and all timers receive the same
   * initial timestamp and interval, this will sync the
   * devices up with an error quite close to that of the
   * network sync itself.
   */
  struct timespec realtime, monotime;
  clock_gettime(CLOCK_REALTIME, &realtime);
  clock_gettime(CLOCK_MONOTONIC, &monotime);

  auto real_ns = std::chrono::seconds{realtime.tv_sec} + std::chrono::nanoseconds{realtime.tv_nsec};
  auto mono_ns = std::chrono::seconds{monotime.tv_sec} + std::chrono::nanoseconds{monotime.tv_nsec};

  auto target = initial_timestamp + (counter++ * interval);
  auto ns_til_target = target - real_ns;

  if (ns_til_target <= std::chrono::nanoseconds{0}) {
    uint32_t intervals_elapsed = -ns_til_target / interval + 1;
    auto ns_elapsed = intervals_elapsed * interval;
    ns_til_target += ns_elapsed;  // adjust ns till target for arming this timer
    counter += intervals_elapsed; // adjust counter for future timers
    target += ns_elapsed;         // adjust target for return value
  }

  auto mono_target_ns = mono_ns + ns_til_target;

  struct itimerspec its;
  its.it_value.tv_sec = std::chrono::duration_cast<std::chrono::seconds>(mono_target_ns).count();
  its.it_value.tv_nsec = (mono_target_ns - std::chrono::seconds{its.it_value.tv_sec}).count();
  its.it_interval.tv_sec = 0;
  its.it_interval.tv_nsec = 0;

  timer_settime(timerid, TIMER_ABSTIME, &its, nullptr);

  return target;
}
