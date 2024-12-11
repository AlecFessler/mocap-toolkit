#ifndef VIDDEC_H
#define VIDDEC_H

#include <stdint.h>

struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct AVBufferRef;

typedef struct decoder {
  struct AVCodecContext* ctx;
  struct AVFrame* frame;
  struct AVFrame* hw_frame;
  struct AVPacket* pkt;
  struct AVBufferRef* hw_device_ctx;

  uint32_t width;
  uint32_t height;
} decoder;

int init_decoder(
  decoder* dec,
  uint32_t width,
  uint32_t height
);

int decode_packet(
  decoder* dec,
  uint8_t* data,
  uint32_t size
);

int recv_frame(
  decoder* dec,
  uint8_t* out_buf
);

int flush_decoder(decoder* dec);
void cleanup_decoder(decoder* dec);

#endif // VIDDEC_H
