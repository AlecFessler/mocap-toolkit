// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef ENCODER_HPP
#define ENCODER_HPP

#include <cstdint>
extern "C" {
#include <libavcodec/avcodec.h>
}
#include <span>
#include <vector>

class Encoder {
private:
  std::pair<uint32_t, uint32_t> m_resolution;
  uint32_t m_pts;
  const AVCodec* m_codec;
  AVCodecContext* m_ctx;
  AVFrame* m_frame;

public:
  Encoder(
    std::pair<uint32_t, uint32_t> resolution,
    uint32_t fps,
    std::vector<AVPacket*>& packets
  );
  ~Encoder();

  void encode(const std::span<uint8_t> frame, AVPacket* packet);
};

#endif // ENCODER_HPP
