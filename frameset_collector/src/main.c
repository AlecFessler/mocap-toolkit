#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "spsc_queue.h"
#include "logging.h"
#include "parse_conf.h"
#include "stream_mgr.h"
#include "network.h"

#define TIMESTAMP_DELAY 1 // seconds
#define FRAME_BUFS_PER_THREAD 32

int main() {
  int ret = 0;
  char logstr[128];

  // setup logging
  char* log_path = "/var/log/mocap-toolkit/server.log";
  ret = setup_logging(log_path);
  if (ret) {
    printf("Error opening log file: %s\n", strerror(errno));
    return -errno;
  }

  // count cameras in conf
  char* cams_conf_path = "/etc/mocap-toolkit/cams.yaml";
  int cam_count = count_cameras(cams_conf_path);
  if (cam_count <= 0) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error getting camera count: %s",
      strerror(cam_count)
    );
    log(ERROR, logstr);
    cleanup_logging();
    return cam_count;
  }

  struct cam_conf confs[cam_count];

  // parse conf file and populate conf structs
  ret = parse_conf(confs, cam_count);
  if (ret) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error parsing camera confs %s",
      strerror(ret)
    );
    log(ERROR, logstr);
    cleanup_logging();
    return ret;
  }

  // pin to cam_count % 8 to stay on ccd0 for 3dv cache with threads
  // but not be on the same core as any threads until there are 8+
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cam_count % 8, &cpuset);

  pid_t pid = getpid();
  ret = sched_setaffinity(
    pid,
    sizeof(cpu_set_t),
    &cpuset
  );
  if (ret == -1) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error pinning process: %s",
      strerror(errno)
    );
    log(ERROR, logstr);
    cleanup_logging();
    return -errno;
  }

  const uint64_t frame_bufs_count = cam_count * FRAME_BUFS_PER_THREAD;
  struct ts_frame_buf ts_frame_bufs[frame_bufs_count];

  const uint64_t frame_buf_size = DECODED_FRAME_WIDTH * DECODED_FRAME_HEIGHT * 3 / 2;
  uint8_t* frame_bufs = malloc(frame_bufs_count * frame_buf_size);
  if (!frame_bufs) {
    log(ERROR, "Failed to allocate frame buffers");
    cleanup_logging();
    return -ENOMEM;
  }

  for (uint i = 0; i < frame_bufs_count; i++) {
    size_t offset = i * frame_buf_size;
    ts_frame_bufs[i].frame_buf = frame_bufs + offset;
  }

  struct producer_q filled_frame_producer_qs[cam_count];
  struct consumer_q filled_frame_consumer_qs[cam_count];

  struct producer_q empty_frame_producer_qs[cam_count];
  struct consumer_q empty_frame_consumer_qs[cam_count];

  void* q_bufs = aligned_alloc(
    CACHE_LINE_SIZE,
    sizeof(void*) * frame_bufs_count * 2
  );
  if (q_bufs == NULL) {
    log(ERROR, "Failed to allocate queue buffers");
    free(frame_bufs);
    cleanup_logging();
    return -ENOMEM;
  }

  void** q_slots = (void**)q_bufs;
  for (int i = 0; i < cam_count; i++) {
    spsc_queue_init(
      &filled_frame_producer_qs[i],
      &filled_frame_consumer_qs[i],
      q_slots + (i * 2 * FRAME_BUFS_PER_THREAD),
      FRAME_BUFS_PER_THREAD
    );

    spsc_queue_init(
      &empty_frame_producer_qs[i],
      &empty_frame_consumer_qs[i],
      q_slots + ((i * 2 + 1) * FRAME_BUFS_PER_THREAD),
      FRAME_BUFS_PER_THREAD
    );

    for (int j = 0; j < FRAME_BUFS_PER_THREAD; j++) {
      spsc_enqueue(
        &empty_frame_producer_qs[i],
        &ts_frame_bufs[i * FRAME_BUFS_PER_THREAD + j]
      );
    }
  }

  struct thread_ctx ctxs[cam_count];
  pthread_t threads[cam_count];
  for (int i = 0; i < cam_count; i++) {
    ctxs[i].conf = &confs[i];
    ctxs[i].filled_bufs = &filled_frame_producer_qs[i];
    ctxs[i].empty_bufs = &empty_frame_consumer_qs[i];
    ctxs[i].core = i % 8;

    ret = pthread_create(
      &threads[i],
      NULL,
      stream_mgr,
      (void*)&ctxs[i]
    );

    if (ret) {
      log(ERROR, "Error spawning thread");
      free(q_bufs);
      free(frame_bufs);
      cleanup_logging();
      return ret;
    }
  }

  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  uint64_t timestamp = (ts.tv_sec + TIMESTAMP_DELAY) * 1000000000ULL + ts.tv_nsec;
  broadcast_msg(confs, cam_count, (char*)&timestamp, sizeof(timestamp));

  struct ts_frame_buf* current_frames[cam_count];
  memset(current_frames, 0, sizeof(struct ts_frame_buf*) * cam_count);

  // count to 100 just for testing
  uint32_t complete_sets = 0;
  while (complete_sets < 100) {
    // dequeue a full set of timestamped frame buffers from each worker thread
    bool full_set = true;
    for(int i = 0; i < cam_count; i++) {
      if (current_frames[i] != NULL)
        continue; // already set

      current_frames[i] = spsc_dequeue(&filled_frame_consumer_qs[i]);

      if (current_frames[i] == NULL) {
        full_set = false; // queue empty
        continue;
      }
    }

    if (!full_set)
      continue; // need a full set to proceed

    // find the max timestamp
    uint64_t max_timestamp = 0;
    for (int i = 0; i < cam_count; i++) {
      if (current_frames[i]->timestamp > max_timestamp)
        max_timestamp = current_frames[i]->timestamp;
    }

    // check if all have matching timestamps
    bool all_equal = true;
    for (int i = 0; i < cam_count; i++) {
      if (current_frames[i]->timestamp != max_timestamp) {
        all_equal = false;
        spsc_enqueue(&empty_frame_producer_qs[i], current_frames[i]);
        current_frames[i] = NULL; // get a new timestamped buffer
      }
    }

    if (!all_equal)
      continue;

    snprintf(
      logstr,
      sizeof(logstr),
      "Received full frame set %d with timestamp %lu",
      complete_sets,
      max_timestamp
    );
    log(INFO, logstr);

    // get a new full set
    for (int i = 0; i < cam_count; i++) {
      spsc_enqueue(&empty_frame_producer_qs[i], current_frames[i]);
      current_frames[i] = NULL;
    }

    complete_sets++;
  }

  const char* stop_msg = "STOP";
  broadcast_msg(confs, cam_count, stop_msg, strlen(stop_msg));

  for (int i = 0; i < cam_count; i++) {
    pthread_join(threads[i], NULL);
  }

  free(q_bufs);
  free(frame_bufs);
  cleanup_logging();
  return ret;
}
