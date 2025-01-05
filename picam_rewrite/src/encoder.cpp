// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>

#include "encoder.hpp"
#include "logging.hpp"

Encoder::Encoder(
  std::pair<uint32_t, uint32_t> resolution,
  uint32_t fps
) :
  m_resolution(resolution),
  m_pts(0) {

  m_codec = avcodec_find_encoder_by_name("libx264");
  if (m_codec == nullptr) {
    std::string err_msg = "Could not find libx264 encoder";
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  m_ctx = avcodec_alloc_context3(m_codec);
  if (m_ctx == nullptr) {
    std::string err_msg = "Failed to allocate encoder context";
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  m_ctx->width = resolution.first;
  m_ctx->height = resolution.second;
  m_ctx->time_base = AVRational{1, 90000}; // 90KHz
  m_ctx->framerate = AVRational{static_cast<int32_t>(fps), 1};
  m_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
  m_ctx->codec_type = AVMEDIA_TYPE_VIDEO;

  AVDictionary *opts = nullptr;
  av_dict_set(&opts, "preset", "ultrafast", 0);
  av_dict_set(&opts, "tune", "zerolatency", 0);

  int status = avcodec_open2(
    m_ctx,
    m_codec,
    &opts
  );
  if (status < 0) {
    av_dict_free(&opts);
    avcodec_free_context(&m_ctx);
    std::string err_msg =
      "Could not open codec: "
      + std::string(strerror(status));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  av_dict_free(&opts);

  m_frame = av_frame_alloc();
  if (m_frame == nullptr) {
    avcodec_free_context(&m_ctx);
    std::string err_msg = "Failed to allocate frame for encoder";
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  m_frame->format = m_ctx->pix_fmt;
  m_frame->width = m_ctx->width;
  m_frame->height = m_ctx->height;

  status = av_frame_get_buffer(m_frame, 0);
  if (status < 0) {
    av_frame_free(&m_frame);
    avcodec_free_context(&m_ctx);
    std::string err_msg =
      "Failed to allocate frame buffer for encoder: "
      + std::string(strerror(status));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  m_pkt = av_packet_alloc();
  if (m_pkt == nullptr) {
    av_frame_free(&m_frame);
    avcodec_free_context(&m_ctx);
    std::string err_msg = "Failed to allocate packet for encoder";
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }
}

Encoder::Encoder(Encoder&& other) noexcept :
  m_resolution(other.m_resolution),
  m_pts(other.m_pts),
  m_codec(other.m_codec),
  m_ctx(other.m_ctx),
  m_frame(other.m_frame),
  m_pkt(other.m_pkt) {

  other.m_ctx = nullptr;
  other.m_frame = nullptr;
  other.m_pkt = nullptr;
}

Encoder::~Encoder() {
  if (m_pkt != nullptr)
    av_packet_free(&m_pkt);
  if (m_frame != nullptr)
    av_frame_free(&m_frame);
  if (m_ctx != nullptr)
    avcodec_free_context(&m_ctx);
}

std::span<uint8_t> Encoder::encode(const std::span<uint8_t>& frame) {
  int status = av_frame_make_writable(m_frame);
  if (status < 0) {
    std::string err_msg =
      "Failed to make frame writable in encoder: "
      + std::string(strerror(status));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }
  const uint32_t y_plane_size = m_resolution.first * m_resolution.second;
  const uint32_t uv_plane_size = y_plane_size / 4;

  m_frame->data[0] = frame.data();
  m_frame->data[1] = frame.data() + y_plane_size;
  m_frame->data[2] = frame.data() + y_plane_size + uv_plane_size;
  m_frame->pts = m_pts++;

  //log_(BENCHMARK, "Starting encoding...");

  status = avcodec_send_frame(m_ctx, m_frame);
  if (status < 0) {
    std::string err_msg =
      "Error encoding frame: "
      + std::string(strerror(status));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  status = avcodec_receive_packet(m_ctx, m_pkt);
  if (status < 0) {
    std::string err_msg =
      "Error receiving encoder packet: "
      + std::string(strerror(status));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  //log_(BENCHMARK, "Encoded frame");

  return std::span(m_pkt->data, m_pkt->size);
}
