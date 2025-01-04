#ifndef VID_PLAYER_HPP
#define VID_PLAYER_HPP

#include <opencv2/opencv.hpp>
#include <spsc_queue.hpp>
#include <sys/types.h>

class VidPlayer {
private:
  timer_t timerid;

public:
  VidPlayer(
    struct consumer_q& frame_queue,
    uint32_t fps
  );
  ~VidPlayer();
}

#endif // VID_PLAYER_HPP
