#include "x264enc.h"
#include <stdexcept>

x264_encoder::x264_encoder(const config_parser& config)
    : width(config.get_int("FRAME_WIDTH")),
      height(config.get_int("FRAME_HEIGHT")) {

    codec = avcodec_find_encoder_by_name("libx264");
    if (!codec) {
        throw std::runtime_error("Could not find libx264 encoder");
    }

    ctx = avcodec_alloc_context3(codec);
    if (!ctx) {
        throw std::runtime_error("Could not allocate encoder context");
    }

    ctx->width = width;
    ctx->height = height;
    ctx->time_base = AVRational{1, config.get_int("FPS")};
    ctx->framerate = AVRational{config.get_int("FPS"), 1};
    ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    ctx->codec_type = AVMEDIA_TYPE_VIDEO;

    AVDictionary *opts = NULL;
    av_dict_set(&opts, "preset", config.get_string("ENC_SPEED").c_str(), 0);
    av_dict_set(&opts, "crf", std::to_string(config.get_int("ENC_QUALITY")).c_str(), 0);

    if (avcodec_open2(ctx, codec, &opts) < 0) {
        av_dict_free(&opts);
        avcodec_free_context(&ctx);
        throw std::runtime_error("Could not open codec");
    }
    av_dict_free(&opts);

    frame = av_frame_alloc();
    if (!frame) {
        avcodec_free_context(&ctx);
        throw std::runtime_error("Could not allocate frame");
    }

    frame->format = ctx->pix_fmt;
    frame->width = ctx->width;
    frame->height = ctx->height;

    if (av_frame_get_buffer(frame, 0) < 0) {
        av_frame_free(&frame);
        avcodec_free_context(&ctx);
        throw std::runtime_error("Could not allocate frame data");
    }

    pkt = av_packet_alloc();
    if (!pkt) {
        av_frame_free(&frame);
        avcodec_free_context(&ctx);
        throw std::runtime_error("Could not allocate packet");
    }
}

void x264_encoder::encode_frame(uint8_t* yuv420_data) {
    if (pkt) av_packet_unref(pkt);

    const int y_size = width * height;
    const int uv_size = y_size / 4;

    av_frame_make_writable(frame);

    memcpy(frame->data[0], yuv420_data, y_size);
    memcpy(frame->data[1], yuv420_data + y_size, uv_size);
    memcpy(frame->data[2], yuv420_data + y_size + uv_size, uv_size);

    if (avcodec_send_frame(ctx, frame) < 0)
        throw std::runtime_error("Error sending frame for encoding");

    int ret = avcodec_receive_packet(ctx, pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
        pkt->size = 0;
    else if (ret < 0)
        throw std::runtime_error("Error receiving encoded packet");
}

x264_encoder::~x264_encoder() {
    if (pkt) av_packet_free(&pkt);
    if (frame) av_frame_free(&frame);
    if (ctx) avcodec_free_context(&ctx);
}
