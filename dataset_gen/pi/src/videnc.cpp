// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <functional>
#include <stdexcept>
#include <string>
#include "logger.h"
#include "videnc.h"

extern std::unique_ptr<logger_t> logger;

videnc::videnc(const config& config)
  : width(config.frame_width),
    height(config.frame_height),
    pts_counter(0) {

  codec = avcodec_find_encoder_by_name("libx264");
  if (!codec) {
    const char* err = "Could not find libx264 encoder";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  ctx = avcodec_alloc_context3(codec);
  if (!ctx) {
    const char* err = "Could not allocate encoder context";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  ctx->width = width;
  ctx->height = height;
  ctx->time_base = AVRational{1, config.fps};
  ctx->framerate = AVRational{config.fps, 1};
  ctx->pix_fmt = AV_PIX_FMT_YUV420P;
  ctx->codec_type = AVMEDIA_TYPE_VIDEO;

  AVDictionary *opts = NULL;
  av_dict_set(&opts, "preset", config.enc_speed.c_str(), 0);
  av_dict_set(&opts, "crf", config.enc_quality.c_str(), 0);

  if (avcodec_open2(ctx, codec, &opts) < 0) {
    av_dict_free(&opts);
    avcodec_free_context(&ctx);
    const char* err = "Could not open codec";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }
  av_dict_free(&opts);

  frame = av_frame_alloc();
  if (!frame) {
    avcodec_free_context(&ctx);
    const char* err = "Could not allocate frame";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  frame->format = ctx->pix_fmt;
  frame->width = ctx->width;
  frame->height = ctx->height;

  if (av_frame_get_buffer(frame, 0) < 0) {
    av_frame_free(&frame);
    avcodec_free_context(&ctx);
    const char* err = "Could not allocate frame data";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  pkt = av_packet_alloc();
  if (!pkt) {
    av_frame_free(&frame);
    avcodec_free_context(&ctx);
    const char* err = "Could not allocate packet";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }
}

void videnc::encode_frame(uint8_t* yuv420_data, pkt_callback cb, connection& conn) {
  const int y_size = width * height;
  const int uv_size = y_size / 4;

  if (av_frame_make_writable(frame) < 0) {
    const char* err = "Frame is not writable";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  memcpy(frame->data[0], yuv420_data, y_size);
  memcpy(frame->data[1], yuv420_data + y_size, uv_size);
  memcpy(frame->data[2], yuv420_data + y_size + uv_size, uv_size);

  frame->pts = pts_counter++;

  if (avcodec_send_frame(ctx, frame) < 0) {
    const char* err = "Error sending frame for encoding";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  while (true) {
    int ret = avcodec_receive_packet(ctx, pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
      break;
    if (ret < 0) {
      const char* err = "Error receiving encoded packet";
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
      throw std::runtime_error(err);
    }

    if (cb(conn, pkt->data, pkt->size) < 0) {
      const char* err = "Error in stream callback";
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
      throw std::runtime_error(err);
    }

    av_packet_unref(pkt);
  }
}

videnc::~videnc() {
  if (pkt) av_packet_free(&pkt);
  if (frame) av_frame_free(&frame);
  if (ctx) avcodec_free_context(&ctx);
}
