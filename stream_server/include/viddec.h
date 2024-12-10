#ifndef VIDDEC_H
#define VIDDEC_H

#include <stdint.h>

struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct AVBufferRef;

struct decoder {
  struct AVCodecContext* ctx;
  struct AVFrame* frame;
  struct AVFrame* hw_frame;
  struct AVPacket* pkt;
  struct AVBufferRef* hw_device_ctx;

  uint32_t width;
  uint32_t height;
};

int init_decoder(
  struct decoder* dec,
  uint32_t width,
  uint32_t height
);

int decode_packet(
  struct decoder* dec,
  uint8_t* data,
  uint32_t size
);

int recv_frame(
  struct decoder* dec,
  uint8_t* out_buf
);

int flush_decoder(struct decoder* dec);
void cleanup_decoder(struct decoder* dec);

#endif // VIDDEC_H
