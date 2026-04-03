#ifndef VIDDEC_HPP
#define VIDDEC_HPP

#include <cstdint>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/frame.h>
}

struct decoder {
  AVCodecContext* ctx;
  AVFrame* hw_frame;
  AVPacket* pkt;
  AVBufferRef* hw_device_ctx;

  uint32_t width;
  uint32_t height;
};

int init_decoder(
  decoder* dec,
  uint32_t width,
  uint32_t height,
  uint32_t num_surfaces,
  AVBufferRef* shared_hw_device_ctx
);

int decode_packet(
  decoder* dec,
  uint8_t* data,
  uint32_t size
);

int recv_frame(
  decoder* dec,
  void** dev_ptr
);

int flush_decoder(decoder* dec);
void cleanup_decoder(decoder* dec);

#endif // VIDDEC_HPP
