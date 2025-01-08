// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef INTERVAL_TIMER_HPP
#define INTERVAL_TIMER_HPP

#include <chrono>
#include <cstdint>
#include <sys/types.h>

class IntervalTimer {
private:
  timer_t m_timerid;
  std::chrono::nanoseconds m_initial_timestamp;
  std::chrono::nanoseconds m_interval;
  uint32_t m_counter;

public:
  IntervalTimer(
    std::chrono::nanoseconds initial_timestamp,
    std::chrono::nanoseconds interval,
    int signum
  );
  ~IntervalTimer();
  std::chrono::nanoseconds arm_timer();
};

#endif // INTERVAL_TIMER_HPP
