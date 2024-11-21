#ifndef VIDENC_H
#define VIDENC_H

#include <stdint.h>
#include <memory>
#include "config_parser.h"

extern "C" {
#include <libavcodec/avcodec.h>
}

class videnc {
public:
    videnc(const config_parser& config);
    ~videnc();
    videnc(const videnc&) = delete;
    videnc& operator=(const videnc&) = delete;
    videnc(videnc&&) = delete;
    videnc& operator=(videnc&&) = delete;

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
