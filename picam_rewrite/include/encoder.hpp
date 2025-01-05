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

class Encoder {
private:
  std::pair<uint32_t, uint32_t> m_resolution;
  uint32_t m_pts;
  const AVCodec* m_codec;
  AVCodecContext* m_ctx;
  AVFrame* m_frame;
  AVPacket* m_pkt;

public:
  Encoder(
    std::pair<uint32_t, uint32_t> resolution,
    uint32_t fps
  );
  Encoder(Encoder&& other) noexcept;
  ~Encoder();

  std::span<uint8_t> encode(const std::span<uint8_t>& frame);
};

#endif // ENCODER_HPP
