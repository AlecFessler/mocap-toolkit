#include "viddec.h"
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/frame.h>
#include <string.h>

int init_decoder(
  struct decoder* dec,
  uint32_t width,
  uint32_t height
) {
  int ret = 0;

  // Initialize structure
  memset(dec, 0, sizeof(*dec));
  dec->width = width;
  dec->height = height;

  // Find h264 NVDEC decoder
  const AVCodec* codec = avcodec_find_decoder_by_name("h264_nvdec");
  if (!codec) {
    fprintf(stderr, "Could not find NVDEC H.264 decoder\n");
    return -1;
  }

  // Create decoder context
  dec->ctx = avcodec_alloc_context3(codec);
  if (!dec->ctx) {
    fprintf(stderr, "Could not allocate decoder context\n");
    return -1;
  }

  // Create CUDA device context
  ret = av_hwdevice_ctx_create(
    &dec->hw_device_ctx,
    AV_HWDEVICE_TYPE_CUDA,
    NULL,
    NULL,
    0
  );
  if (ret < 0) {
    fprintf(stderr, "Failed to create CUDA device: %d\n", ret);
    goto cleanup;
  }

  // Set hardware frame context
  dec->ctx->hw_device_ctx = av_buffer_ref(dec->hw_device_ctx);
  if (!dec->ctx->hw_device_ctx) {
    fprintf(stderr, "Failed to reference hw device context\n");
    goto cleanup;
  }

  // Set basic parameters
  dec->ctx->width = width;
  dec->ctx->height = height;
  dec->ctx->pix_fmt = AV_PIX_FMT_CUDA;

  // Open decoder
  ret = avcodec_open2(
    dec->ctx,
    codec,
    NULL
  );
  if (ret < 0) {
    fprintf(stderr, "Failed to open codec: %d\n", ret);
    goto cleanup;
  }

  // Allocate frames and packet
  dec->frame = av_frame_alloc();
  dec->hw_frame = av_frame_alloc();
  dec->pkt = av_packet_alloc();
  if (!dec->frame || !dec->hw_frame || !dec->pkt) {
    fprintf(stderr, "Failed to allocate frame/packet\n");
    goto cleanup;
  }

  // Set up CPU frame properties and buffer
  dec->frame->format = AV_PIX_FMT_NV12;
  dec->frame->width = width;
  dec->frame->height = height;
  ret = av_frame_get_buffer(dec->frame, 0);
  if (ret < 0) {
    fprintf(stderr, "Failed to allocate frame buffer\n");
    goto cleanup;
  }

  return 0;

cleanup:
  cleanup_decoder(dec);
  return ret;
}

void cleanup_decoder(struct decoder* dec) {
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

int decode_packet(struct decoder* dec, uint8_t* data, uint32_t size) {
  dec->pkt->data = data;
  dec->pkt->size = size;

  int ret = avcodec_send_packet(dec->ctx, dec->pkt);
  if (ret < 0) {
    fprintf(stderr, "Error sending packet for decoding\n");
    return ret;
  }

  return 0;
}

int receive_frame(struct decoder* dec, uint8_t* out_buf) {
  int ret = 0;

  // First receive into the hardware frame
  ret = avcodec_receive_frame(dec->ctx, dec->hw_frame);
  if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
    return ret;
  } else if (ret < 0) {
    fprintf(stderr, "Error receiving frame from decoder\n");
    return ret;
  }

  // Transfer from GPU to CPU memory
  ret = av_hwframe_transfer_data(dec->frame, dec->hw_frame, 0);
  if (ret < 0) {
    fprintf(stderr, "Error transferring frame from GPU to CPU\n");
    return ret;
  }

  const size_t frame_size = dec->width * dec->height * 3 / 2;
  memcpy(out_buf, dec->frame->data[0], frame_size);

  return 0;
}

int flush_decoder(struct decoder* dec) {
  int ret = avcodec_send_packet(dec->ctx, NULL);
  if (ret < 0) {
    fprintf(stderr, "Error flushing decoder\n");
    return ret;
  }
  return 0;
}
