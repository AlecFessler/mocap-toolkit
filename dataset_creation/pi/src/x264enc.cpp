#include "x264enc.h"
#include <stdexcept>

x264_encoder::x264_encoder(const config_parser& config)
    : width(config.get_int("FRAME_WIDTH")),
      height(config.get_int("FRAME_HEIGHT")) {

    // Find the x264 encoder
    codec = avcodec_find_encoder_by_name("libx264");
    if (!codec) {
        throw std::runtime_error("Could not find libx264 encoder");
    }

    // Allocate context
    ctx = avcodec_alloc_context3(codec);
    if (!ctx) {
        throw std::runtime_error("Could not allocate encoder context");
    }

    // Setup basic parameters
    ctx->width = width;
    ctx->height = height;
    ctx->time_base = AVRational{1, config.get_int("FPS")};
    ctx->framerate = AVRational{config.get_int("FPS"), 1};
    ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    ctx->codec_type = AVMEDIA_TYPE_VIDEO;

    AVDictionary *opts = NULL;
    av_dict_set(&opts, "preset", config.get_string("ENC_SPEED").c_str(), 0);
    av_dict_set(&opts, "crf", std::to_string(config.get_int("ENC_QUALITY")).c_str(), 0);

    // Open codec with options dictionary
    if (avcodec_open2(ctx, codec, &opts) < 0) {
        av_dict_free(&opts);
        avcodec_free_context(&ctx);
        throw std::runtime_error("Could not open codec");
    }
    av_dict_free(&opts);

    // Allocate frame and packet
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

    // Allocate output buffer (same size as before)
    out_buf = std::make_unique<uint8_t[]>(width * height * 2);
}

size_t x264_encoder::encode_frame(uint8_t* yuv420_data) {
    const int y_size = width * height;
    const int uv_size = y_size / 4;

    // Make sure the frame is writable
    av_frame_make_writable(frame);

    // Copy YUV420 data to frame
    memcpy(frame->data[0], yuv420_data, y_size);
    memcpy(frame->data[1], yuv420_data + y_size, uv_size);
    memcpy(frame->data[2], yuv420_data + y_size + uv_size, uv_size);

    // Send frame to encoder
    if (avcodec_send_frame(ctx, frame) < 0) {
        throw std::runtime_error("Error sending frame for encoding");
    }

    // Get encoded packet
    int ret = avcodec_receive_packet(ctx, pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        return 0;
    } else if (ret < 0) {
        throw std::runtime_error("Error receiving encoded packet");
    }

    // Copy to our output buffer
    memcpy(out_buf.get(), pkt->data, pkt->size);
    size_t size = pkt->size;

    // Unref the packet for reuse
    av_packet_unref(pkt);

    return size;
}

x264_encoder::~x264_encoder() {
    if (pkt)
        av_packet_free(&pkt);
    if (frame)
        av_frame_free(&frame);
    if (ctx)
        avcodec_free_context(&ctx);
}
