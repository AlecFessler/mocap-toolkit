#ifndef PIPELINE_CONFIG_H
#define PIPELINE_CONFIG_H

#include <atomic>
#include <cstdint>

struct pipeline_config {
  // stats generation counter (bump to reset running averages)
  std::atomic<uint32_t> stats_generation{0};

  // running average reproj error (written by inference thread, read by renderer)
  std::atomic<float> avg_reproj_running{0.0f};
  std::atomic<int32_t> avg_reproj_frames{0};
};

#endif // PIPELINE_CONFIG_H
