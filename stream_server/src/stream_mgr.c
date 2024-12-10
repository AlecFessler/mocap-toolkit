#define _GNU_SOURCE
#include <errno.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "logging.h"
#include "network.h"
#include "stream_mgr.h"

void* stream_mgr(void* ptr) {
  char logstr[128];

  struct thread_ctx* ctx = (struct thread_ctx*)ptr;
  uint8_t enc_frame_buf[ENCODED_FRAME_BUF_SIZE];

  pin_to_core(ctx->core);

  int sockfd = setup_stream(ctx->conf);
  if (sockfd < 0) {
    return NULL;
  }

  int clientfd = accept_conn(sockfd);

  struct __attribute__((packed)) header {
    uint64_t timestamp;
    uint32_t size;
  };
  struct header hdr;

  ssize_t pkt_size = 0;
  do {
    // reset header
    memset(&hdr, 0, sizeof(hdr));

    // receive header
    pkt_size = recv_from_stream(
      clientfd,
      (char*)&hdr,
      sizeof(hdr)
    );

    // validate packet size
    if (pkt_size != sizeof(hdr)) {
      snprintf(
        logstr,
        sizeof(logstr),
        "Received unexpected header size with %zd bytes from cam %s",
        pkt_size,
        ctx->conf->name
      );
      log(ERROR, logstr);
    }

    // check if end of stream
    if (memcmp(&hdr.timestamp, "EOSTREAM", 8) == 0) {
      log(INFO, "Received end of stream signal");
      break;
    }

    // validate frame size
    if (hdr.size > ENCODED_FRAME_BUF_SIZE) {
      snprintf(
        logstr,
        sizeof(logstr),
        "Received frame size that is larger than the allocated buffer of %d bytes: %d",
        ENCODED_FRAME_BUF_SIZE,
        hdr.size
      );
      log(ERROR, logstr);
      break;
    }

    // receive frame
    pkt_size = recv_from_stream(
      clientfd,
      (char*)enc_frame_buf,
      hdr.size
    );

    // validate packet size
    if (pkt_size != hdr.size) {
      snprintf(
        logstr,
        sizeof(logstr),
        "Received unexpected frame size with %zd bytes from cam %s",
        pkt_size,
        ctx->conf->name
      );
      log(ERROR, logstr);
    }

    snprintf(
      logstr,
      sizeof(logstr),
      "Received frame with %zd bytes from cam %s with timestamp %lu",
      pkt_size,
      ctx->conf->name,
      hdr.timestamp
    );
    log(INFO, logstr);

  } while(pkt_size > 0);

  if (sockfd >= 0) {
    close(sockfd);
  }
  if (clientfd >= 0) {
    close(clientfd);
  }
  return NULL;
}

int pin_to_core(int core) {
  int ret = 0;
  char logstr[128];

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);

  pid_t tid = gettid();
  ret = sched_setaffinity(
    tid,
    sizeof(cpu_set_t),
    &cpuset
  );

  if (ret == -1) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error pinning thread %d to core %d, err: %s",
      tid,
      core,
      strerror(errno)
    );
    log(ERROR, logstr);
    return -errno;
  }

  snprintf(
    logstr,
    sizeof(logstr),
    "Successfuly pinned thread %d to core %d",
    tid,
    core
  );
  log(INFO, logstr);

  return ret;
}
