#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/frame.h>
#include <string.h>

#include "logging.h"
#include "viddec.h"

int init_decoder(
  decoder* dec,
  uint32_t width,
  uint32_t height,
  uint32_t num_surfaces
) {
  int ret = 0;
  char logstr[128];

  memset(dec, 0, sizeof(*dec));
  dec->width = width;
  dec->height = height;

  const AVCodec* codec = avcodec_find_decoder_by_name("h264_cuvid");
  if (!codec) {
    log(ERROR, "Could not find cuvid H.264 decoder");
    return -ENODEV;
  }

  dec->ctx = avcodec_alloc_context3(codec);
  if (!dec->ctx) {
    log(ERROR, "Could not allocate decoder context");
    return -ENOMEM;
  }

  ret = av_hwdevice_ctx_create(
    &dec->hw_device_ctx,
    AV_HWDEVICE_TYPE_CUDA,
    NULL,
    NULL,
    0
  );
  if (ret < 0) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Failed to create CUDA device: %s",
      strerror(ret)
    );
    log(ERROR, logstr);
    goto cleanup;
  }

  dec->ctx->hw_device_ctx = av_buffer_ref(dec->hw_device_ctx);
  if (!dec->ctx->hw_device_ctx) {
    log(ERROR, "Failed to reference hw device context");
    goto cleanup;
  }

  dec->ctx->pix_fmt = AV_PIX_FMT_NONE;
  dec->ctx->pkt_timebase = (AVRational){1, 90000}; // 90 KHz
  dec->ctx->flags |= AV_CODEC_FLAG_LOW_DELAY;

  AVDictionary* opts = NULL;
  char num_surfaces_str[4];
  snprintf(
    num_surfaces_str,
    sizeof(num_surfaces_str),
    "%d",
    num_surfaces
  );
  av_dict_set(&opts, "surfaces", num_surfaces_str, 0);

  ret = avcodec_open2(
    dec->ctx,
    codec,
    &opts
  );
  if (ret < 0) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Failed to open codec: %s",
      strerror(ret)
    );
    goto cleanup;
  }

  dec->hw_frame = av_frame_alloc();
  dec->pkt = av_packet_alloc();
  if (!dec->hw_frame || !dec->pkt) {
    log(ERROR, "Failed to allocate frame/packet");
    goto cleanup;
  }

  return 0;

  cleanup:
  cleanup_decoder(dec);
  return ret;
}

void cleanup_decoder(decoder* dec) {
  if (dec->pkt) {
    av_packet_free(&dec->pkt);
  }
  if (dec->hw_frame) {
    av_frame_free(&dec->hw_frame);
  }
  if (dec->ctx) {
    avcodec_free_context(&dec->ctx);
  }
  if (dec->hw_device_ctx) {
    av_buffer_unref(&dec->hw_device_ctx);
  }
}

int decode_packet(decoder* dec, uint8_t* data, uint32_t size) {
  char logstr[128];

  dec->pkt->data = data;
  dec->pkt->size = size;

  int ret = avcodec_send_packet(dec->ctx, dec->pkt);
  if (ret < 0) {
    char err[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(ret, err, AV_ERROR_MAX_STRING_SIZE);
    snprintf(
      logstr,
      sizeof(logstr),
      "Error sending packet for decoding: %s",
      err
    );
    log(ERROR, logstr);
    return ret;
  }

  return 0;
}

int recv_frame(decoder* dec, void** dev_ptr) {
  int ret = 0;

  ret = avcodec_receive_frame(dec->ctx, dec->hw_frame);
  if (ret == AVERROR(EAGAIN)) {
    return EAGAIN; // need more frames
  } else if (ret == AVERROR_EOF) {
    return ENODATA; // end of stream
  } else if (ret < 0) {
    log(ERROR, "Error receiving frame from decoder");
    return ret;
  }

  *dev_ptr = dec->hw_frame->data[0];
  return 0;
}

int flush_decoder(decoder* dec) {
  int ret = avcodec_send_packet(dec->ctx, NULL);
  if (ret < 0) {
    log(ERROR, "Error flushing decoder");
    return ret;
  }

  return 0;
}
