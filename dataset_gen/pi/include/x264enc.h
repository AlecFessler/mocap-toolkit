#ifndef X264ENC_H
#define X264ENC_H

#include <stdint.h>
#include <memory>
#include "config_parser.h"

extern "C" {
#include <libavcodec/avcodec.h>
}

class x264_encoder {
public:
    x264_encoder(const config_parser& config);
    ~x264_encoder();
    x264_encoder(const x264_encoder&) = delete;
    x264_encoder& operator=(const x264_encoder&) = delete;
    x264_encoder(x264_encoder&&) = delete;
    x264_encoder& operator=(x264_encoder&&) = delete;

    void encode_frame(uint8_t* yuv420_data);
    AVPacket* pkt;

private:
    int width;
    int height;
    const AVCodec* codec;
    AVCodecContext* ctx;
    AVFrame* frame;
};
#endif
