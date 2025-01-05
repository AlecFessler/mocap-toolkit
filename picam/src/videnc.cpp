// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <functional>
#include <stdexcept>
#include <string>

#include "logging.h"
#include "videnc.h"

videnc::videnc(const config& config)
  : width(config.frame_width),
    height(config.frame_height),
    pts_counter(0) {
  codec = avcodec_find_encoder_by_name("libx264");
  if (!codec) {
    const char* err = "Could not find libx264 encoder";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }

  ctx = avcodec_alloc_context3(codec);
  if (!ctx) {
    const char* err = "Could not allocate encoder context";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }

  ctx->width = width;
  ctx->height = height;
  ctx->time_base = AVRational{1, 90000}; // 90KHz
  ctx->framerate = AVRational{config.fps, 1};
  ctx->pix_fmt = AV_PIX_FMT_YUV420P;
  ctx->codec_type = AVMEDIA_TYPE_VIDEO;

  AVDictionary *opts = NULL;
  av_dict_set(&opts, "preset", config.enc_speed.c_str(), 0);
  av_dict_set(&opts, "crf", config.enc_quality.c_str(), 0);
  av_dict_set(&opts, "tune", "zerolatency", 0);

  if (avcodec_open2(ctx, codec, &opts) < 0) {
    av_dict_free(&opts);
    avcodec_free_context(&ctx);
    const char* err = "Could not open codec";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }
  av_dict_free(&opts);

  frame = av_frame_alloc();
  if (!frame) {
    avcodec_free_context(&ctx);
    const char* err = "Could not allocate frame";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }

  frame->format = ctx->pix_fmt;
  frame->width = ctx->width;
  frame->height = ctx->height;

  if (av_frame_get_buffer(frame, 0) < 0) {
    av_frame_free(&frame);
    avcodec_free_context(&ctx);
    const char* err = "Could not allocate frame data";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }

  pkt = av_packet_alloc();
  if (!pkt) {
    av_frame_free(&frame);
    avcodec_free_context(&ctx);
    const char* err = "Could not allocate packet";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }
}

videnc::~videnc() {
  if (pkt) av_packet_free(&pkt);
  if (frame) av_frame_free(&frame);
  if (ctx) avcodec_free_context(&ctx);
}

void videnc::encode_frame(uint8_t* data) {
  const int y_size = width * height;
  const int uv_size = y_size / 4;

  if (av_frame_make_writable(frame) < 0) {
    const char* err = "Frame is not writable";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }

  frame->data[0] = data;
  frame->data[1] = data + y_size;
  frame->data[2] = data + y_size + uv_size;

  frame->pts = pts_counter++;

  if (avcodec_send_frame(ctx, frame) < 0) {
    const char* err = "Error sending frame for encoding";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }
}

void videnc::flush() {
  int ret = avcodec_send_frame(ctx, nullptr); // signal end of stream
  if (ret < 0) {
    const char* err = "Error signaling EOF to encoder";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }
}

uint8_t* videnc::recv_frame(int& size) {
  int ret = avcodec_receive_packet(ctx, pkt);
  if (ret == AVERROR(EAGAIN)) return nullptr; // no packets available yet
  if (ret == AVERROR_EOF) return nullptr; // no more packets
  if (ret < 0) {
    const char* err = "Error receiving packet";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }
  size = pkt->size;
  return pkt->data;
}
