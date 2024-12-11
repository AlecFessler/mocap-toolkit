#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/frame.h>
#include <string.h>

#include "logging.h"
#include "viddec.h"

int init_decoder(
  decoder* dec,
  uint32_t width,
  uint32_t height
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

  dec->ctx->width = width;
  dec->ctx->height = height;
  dec->ctx->pix_fmt = AV_PIX_FMT_CUDA;
  dec->ctx->pkt_timebase = (AVRational){1, 90000};

  ret = avcodec_open2(
    dec->ctx,
    codec,
    NULL
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

  dec->frame = av_frame_alloc();
  dec->hw_frame = av_frame_alloc();
  dec->pkt = av_packet_alloc();
  if (!dec->frame || !dec->hw_frame || !dec->pkt) {
    log(ERROR, "Failed to allocate frame/packet");
    goto cleanup;
  }

  dec->frame->format = AV_PIX_FMT_NV12;
  dec->frame->width = width;
  dec->frame->height = height;
  ret = av_frame_get_buffer(dec->frame, 0);
  if (ret < 0) {
    log(ERROR, "Failed to allocate frame buffer");
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
  if (dec->frame) {
    av_frame_free(&dec->frame);
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
  dec->pkt->data = data;
  dec->pkt->size = size;

  int ret = avcodec_send_packet(dec->ctx, dec->pkt);
  if (ret < 0) {
    log(ERROR, "Error sending packet for decoding");
    return ret;
  }

  return 0;
}

int recv_frame(decoder* dec, uint8_t* out_buf) {
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

  dec->frame->data[0] = out_buf;
  dec->frame->data[1] = out_buf + (dec->width * dec->height);
  dec->frame->linesize[0] = dec->width;
  dec->frame->linesize[1] = dec->width;

  ret = av_hwframe_transfer_data(dec->frame, dec->hw_frame, 0);
  if (ret < 0) {
    log(ERROR, "Error transferring frame from GPU to CPU");
    return ret;
  }

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
