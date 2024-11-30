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
  /**
   * Initializes an H.264 video encoder using libavcodec.
   *
   * Creates a complete encoding pipeline with these steps:
   * 1. Locates the x264 encoder
   * 2. Allocates and configures encoding context
   * 3. Sets up frame format for YUV420 input
   * 4. Initializes encoder with quality/speed settings
   *
   * The encoder is configured for streaming:
   * - YUV420 pixel format matches camera output
   * - Time base and framerate from config ensure proper timing
   * - CRF (Constant Rate Factor) for quality-based bitrate
   * - Preset controls encoding speed/compression tradeoff
   *
   * Parameters:
   *   config: Contains resolution, framerate, and encoding settings
   *
   * Throws:
   *   std::runtime_error: On any initialization failure, with cleanup
   *                      of previously allocated resources
   */
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

videnc::~videnc() {
  /**
   * Releases encoder resources in correct order.
   *
   * Cleanup sequence:
   * 1. Free packet buffer
   * 2. Free frame buffer
   * 3. Free encoder context
   *
   * Note: Each step checks for null before freeing,
   * allowing partial cleanup if constructor fails
   */
  if (pkt) av_packet_free(&pkt);
  if (frame) av_frame_free(&frame);
  if (ctx) avcodec_free_context(&ctx);
}

void videnc::encode_frame(uint8_t* yuv420_data, connection& conn) {
  /**
   * Encodes a single YUV420 frame and streams via callback.
   *
   * The encoding process:
   * 1. Maps DMA buffer YUV planes to encoder frame buffer
   * 2. Assigns presentation timestamp
   * 3. Submits frame for encoding
   * 4. Retrieves and streams any ready packets
   *
   * Note: The encoder may not output a packet for every input
   * frame due to B-frames and rate control. The callback is
   * only invoked when compressed packets are ready.
   *
   * Parameters:
   *   yuv420_data: Raw frame data in YUV420 format
   *   conn: Connection object for streaming packets
   *
   * Throws:
   *   std::runtime_error: If encoding or streaming fails
   */
  const int y_size = width * height;
  const int uv_size = y_size / 4;

  if (av_frame_make_writable(frame) < 0) {
    const char* err = "Frame is not writable";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  frame->data[0] = yuv420_data;
  frame->data[1] = yuv420_data + y_size;
  frame->data[2] = yuv420_data + y_size + uv_size;

  frame->pts = pts_counter++;

  if (avcodec_send_frame(ctx, frame) < 0) {
    const char* err = "Error sending frame for encoding";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  while (true) {
    int ret = avcodec_receive_packet(ctx, pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
    if (ret < 0) {
      const char* err = "Error receiving encoded packet";
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
      throw std::runtime_error(err);
    }

    if (conn.stream_pkt(pkt->data, pkt->size) < 0) {
      const char* err = "Error in stream callback";
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
      throw std::runtime_error(err);
    }

    av_packet_unref(pkt);
  }
}

void videnc::flush(connection& conn) {
  /**
   * Flushes all remaining frames from encoder and streams them
   *
   * When encoding is finished, some frames may still be buffered
   * in the encoder due to B-frames and rate control. This function:
   * 1. Signals end-of-stream to encoder
   * 2. Retrieves all remaining packets
   * 3. Streams them via callback
   *
   * Parameters:
   *   conn: Connection object for streaming packets
   *
   * Throws:
   *   std::runtime_error: If flushing or streaming fails
   */
  int ret = avcodec_send_frame(ctx, nullptr); // signal end of stream
  if (ret < 0) {
    const char* err = "Error signaling EOF to encoder";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  while (true) {
    ret = avcodec_receive_packet(ctx, pkt);
    if (ret == AVERROR_EOF) break; // no more packets
    if (ret < 0) {
      const char* err = "Error receiving packet during flush";
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
      throw std::runtime_error(err);
    }

    if (conn.stream_pkt(pkt->data, pkt->size) < 0) {
      const char* err = "Error in stream callback during flush";
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
      throw std::runtime_error(err);
    }

    av_packet_unref(pkt);
  }
}
