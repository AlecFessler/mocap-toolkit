#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "logging.h"
#include "parse_conf.h"
#include "stream_mgr.h"
#include "network.h"

#define TIMESTAMP_DELAY 1 // seconds

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
    return -errno;
  }

  // initialize data shared between main and spawned threads
  atomic_uint_least32_t frames_filled = ATOMIC_VAR_INIT(0);
  atomic_bool new_frame_flags[cam_count];
  for (int i = 0; i < cam_count; i++) {
    atomic_init(&new_frame_flags[i], true);
  }

  sem_t loop_ctl_sem;
  ret = sem_init(&loop_ctl_sem, 0, 0);
  if (ret) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Failed to initialize semaphore: %s",
      strerror(errno)
    );
    cleanup_logging();
    return -errno;
  }

  uint64_t timestamps[cam_count];

  size_t decoded_frame_size = DECODED_FRAME_HEIGHT * DECODED_FRAME_WIDTH * 1.5;
  uint8_t* frame_bufs = malloc(decoded_frame_size * cam_count);
  if (!frame_bufs) {
    log(ERROR, "Failed to allocate memory for frame buffers");
    sem_destroy(&loop_ctl_sem);
    cleanup_logging();
    return -ENOMEM;
  }

  // spawn threads
  struct thread_ctx ctxs[cam_count];
  pthread_t threads[cam_count];

  for (int i = 0; i < cam_count; i++) {
    ctxs[i].loop_ctl_sem = &loop_ctl_sem;
    ctxs[i].frames_filled = &frames_filled;
    ctxs[i].new_frame = &new_frame_flags[i];
    ctxs[i].frames_total = cam_count;
    ctxs[i].conf = &confs[i];
    ctxs[i].timestamp = &timestamps[i];
    ctxs[i].frame_buf = frame_bufs + (i * decoded_frame_size);
    ctxs[i].core = i % 8;

    ret = pthread_create(
      &threads[i],
      NULL,
      stream_mgr,
      (void*)&ctxs[i]
    );

    if (ret) {
      log(ERROR, "Error spawning thread");
      free(frame_bufs);
      sem_destroy(&loop_ctl_sem);
      cleanup_logging();
      return ret;
    }
  }

  // broadcast timestamp
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  uint64_t timestamp = (ts.tv_sec + TIMESTAMP_DELAY) * 1000000000ULL + ts.tv_nsec;
  broadcast_msg(confs, cam_count, (char*)&timestamp, sizeof(timestamp));

  // remove after debugging
  uint64_t last_timestamp = 0;
  uint32_t completed_frame_sets = 0;
  while (true) {
    sem_wait(&loop_ctl_sem);

    bool all_match = true;
    uint64_t max_timestamp = timestamps[0];
    for (int i = 1; i < cam_count; i++) {
      if (timestamps[i] != max_timestamp) {
        all_match = false;
        if (timestamps[i] < max_timestamp)
          continue;

        max_timestamp = timestamps[i];
        snprintf(
          logstr,
          sizeof(logstr),
          "Found new max timestamp %lu",
          max_timestamp
        );
        log(DEBUG, logstr);
      }
    }

    if (!all_match) {
      for (int i = 0; i < cam_count; i++) {
        if (timestamps[i] < max_timestamp) {
          atomic_fetch_sub(&frames_filled, 1);
          atomic_store(&new_frame_flags[i], true);

          snprintf(
            logstr,
            sizeof(logstr),
            "Cycling thread %d with timestamp %lu, max is %lu",
            i,
            timestamps[i],
            max_timestamp
          );
          log(DEBUG, logstr);
        }
      }
      log(DEBUG, "Cycling through frames...");
      continue;
    }

    log(DEBUG, "All timestamps matched, proceeding...");

    atomic_store(&frames_filled, 0);
    for (int i = 0; i < cam_count; i++) {
      atomic_store(&new_frame_flags[i], true);
    }

    uint64_t delta = max_timestamp - last_timestamp;
    snprintf(
      logstr,
      sizeof(logstr),
      "Timestamp delta: %lu",
      delta
    );
    log(DEBUG, logstr);
    last_timestamp = max_timestamp;

    snprintf(
      logstr,
      sizeof(logstr),
      "Received completed framset with timestamp %lu",
      max_timestamp
    );
    log(INFO, logstr);

    // just for testing
    if (completed_frame_sets++ == 100) {
      log(INFO, "Received 100 completed frame sets");
      break;
    }
  }

  // broadcast stop
  const char* stop_msg = "STOP";
  broadcast_msg(confs, cam_count, stop_msg, strlen(stop_msg));

  for (int i = 0; i < cam_count; i++) {
    pthread_join(threads[i], NULL);
  }

  free(frame_bufs);
  sem_destroy(&loop_ctl_sem);
  cleanup_logging();
  return ret;
}
