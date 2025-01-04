#include <csignal>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <spsc_queue.hpp>
#include <sys/types.h>
#include <time.h>

#include "logging.h"
#include "vid_player.hpp"

constexpr const char* WINDOW_NAME = "Stream Display";
constexpr uint64_t NS_PER_S = 1'000'000'000;

VidPlayer::VidPlayer(
  struct consumer_q& frame_queue,
  uint32_t fps
) {
  struct sigevent sev;
  sev.sigev_notify = SIGEV_SIGNAL;
  sev.sigev_signo = SIGRTMIN;
  sev.sigev_value.sival_ptr = &timerid;
  timer_create(CLOCK_MONOTONIC, &sev, &timerid);

  struct itimerspec its;
  its.it_value.tv_sec = 0;
  its.it_value.tv_nsec = NS_PER_S / fps;
  its.it_interval.tv_sec = 0;
  its.it_interval.tv_nsec = NS_PER_S / fps;
}

VidPlayer::~VidPlayer() {
  
}
