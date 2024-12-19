#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <semaphore.h>
#include <sched.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "spsc_queue.h"
#include "logging.h"
#include "parse_conf.h"
#include "stream_mgr.h"
#include "network.h"

#define LOG_PATH "/var/log/mocap-toolkit/server.log"
#define CAM_CONF_PATH "/etc/mocap-toolkit/cams.yaml"

#define SHM_NAME "/mocap-toolkit_frameset"
#define SEM_CONSUMER_READY "/mocap-toolkit_consumer_ready"
#define SEM_STOP_STREAMS  "/mocap-toolkit_stop_streams"

#define TIMESTAMP_DELAY 1 // seconds
#define FRAME_BUFS_PER_THREAD 32

int main() {
  int ret = 0;
  char logstr[128];

  ret = setup_logging(LOG_PATH);
  if (ret) {
    printf("Error opening log file: %s\n", strerror(errno));
    return -errno;
  }

  int cam_count = count_cameras(CAM_CONF_PATH);
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
  ret = sched_setaffinity(
    getpid(),
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
  const uint64_t frame_buf_size = DECODED_FRAME_WIDTH * DECODED_FRAME_HEIGHT * 3 / 2;

  uint8_t* frame_bufs = malloc(frame_bufs_count * frame_buf_size);
  if (!frame_bufs) {
    log(ERROR, "Failed to allocate frame buffers");
    cleanup_logging();
    return -ENOMEM;
  }

  struct ts_frame_buf ts_frame_bufs[frame_bufs_count];
  for (uint i = 0; i < frame_bufs_count; i++) {
    size_t offset = i * frame_buf_size;
    ts_frame_bufs[i].frame_buf = frame_bufs + offset;
  }

  struct producer_q filled_frame_producer_qs[cam_count];
  struct consumer_q filled_frame_consumer_qs[cam_count];

  struct producer_q empty_frame_producer_qs[cam_count];
  struct consumer_q empty_frame_consumer_qs[cam_count];

  void** q_bufs = aligned_alloc(
    CACHE_LINE_SIZE,
    sizeof(void*) * frame_bufs_count * 2
  );
  if (q_bufs == NULL) {
    log(ERROR, "Failed to allocate queue buffers");
    free(frame_bufs);
    cleanup_logging();
    return -ENOMEM;
  }

  for (int i = 0; i < cam_count; i++) {
    spsc_queue_init(
      &filled_frame_producer_qs[i],
      &filled_frame_consumer_qs[i],
      q_bufs + (i * 2 * FRAME_BUFS_PER_THREAD),
      FRAME_BUFS_PER_THREAD
    );

    spsc_queue_init(
      &empty_frame_producer_qs[i],
      &empty_frame_consumer_qs[i],
      q_bufs + ((i * 2 + 1) * FRAME_BUFS_PER_THREAD),
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

  sem_t* consumer_ready = sem_open(
    SEM_CONSUMER_READY,
    O_CREAT,
    0666,
    0
  );
  if (consumer_ready == SEM_FAILED) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error opening semaphore: %s",
      strerror(errno)
    );
    log(ERROR, logstr);
    free(q_bufs);
    free(frame_bufs);
    cleanup_logging();
    return -errno;
  }

  sem_t* stop_streams = sem_open(
    SEM_STOP_STREAMS,
    O_CREAT,
    0666,
    0
  );
  if (stop_streams == SEM_FAILED) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error opening semaphore: %s",
      strerror(errno)
    );
    log(ERROR, logstr);
    sem_close(consumer_ready);
    sem_unlink(SEM_CONSUMER_READY);
    free(q_bufs);
    free(frame_bufs);
    cleanup_logging();
    return -errno;
  }

  int shm_fd = shm_open(
    SHM_NAME,
    O_CREAT | O_RDWR,
    0666
  );
  if (shm_fd == -1) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error creating shared memory: %s",
      strerror(errno)
    );
    log(ERROR, logstr);
    sem_close(consumer_ready);
    sem_close(stop_streams);
    sem_unlink(SEM_CONSUMER_READY);
    sem_unlink(SEM_STOP_STREAMS);
    free(q_bufs);
    free(frame_bufs);
    cleanup_logging();
  }

  size_t shm_size = frame_buf_size * cam_count;
  ret = ftruncate(
    shm_fd,
    shm_size
  );
  if (ret == -1) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error creating shared memory: %s",
      strerror(errno)
    );
    log(ERROR, logstr);
    close(shm_fd);
    shm_unlink(SHM_NAME);
    sem_close(consumer_ready);
    sem_close(stop_streams);
    sem_unlink(SEM_CONSUMER_READY);
    sem_unlink(SEM_STOP_STREAMS);
    free(q_bufs);
    free(frame_bufs);
    cleanup_logging();
  }

  void* frameset_buf = mmap(
    NULL,
    shm_size,
    PROT_WRITE,
    MAP_SHARED,
    shm_fd,
    0
  );
  if (frameset_buf == MAP_FAILED) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error mapping shared memory: %s",
      strerror(errno)
    );
    log(ERROR, logstr);
    close(shm_fd);
    shm_unlink(SHM_NAME);
    sem_close(consumer_ready);
    sem_close(stop_streams);
    sem_unlink(SEM_CONSUMER_READY);
    sem_unlink(SEM_STOP_STREAMS);
    free(q_bufs);
    free(frame_bufs);
    cleanup_logging();
  }

  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  uint64_t timestamp = (ts.tv_sec + TIMESTAMP_DELAY) * 1000000000ULL + ts.tv_nsec;
  broadcast_msg(confs, cam_count, (char*)&timestamp, sizeof(timestamp));

  struct ts_frame_buf* current_frames[cam_count];
  memset(current_frames, 0, sizeof(struct ts_frame_buf*) * cam_count);

  int stop_val = 0; // set to 1 by frameset consumer process, checked at the end of the loop
  while (!stop_val) {
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
      "Received full frame set with timestamp %lu",
      max_timestamp
    );
    log(INFO, logstr);

    // check if consumer_ready here
    int consumer_ready_val;
    sem_getvalue(consumer_ready, &consumer_ready_val);
    if (consumer_ready_val == 0) { // consumer is waiting on sem
      for (int i = 0; i < cam_count; i++) {
        memcpy(
          frameset_buf + (i * frame_buf_size),
          current_frames[i]->frame_buf,
          frame_buf_size
        );
      }
      log(INFO, "Copied frameset to shared memory for consumer process");
      sem_post(consumer_ready);
    }

    // get a new full set
    for (int i = 0; i < cam_count; i++) {
      spsc_enqueue(&empty_frame_producer_qs[i], current_frames[i]);
      current_frames[i] = NULL;
    }

    // check if we should stop and cleanup
    sem_getvalue(stop_streams, &stop_val);
  }

  const char* stop_msg = "STOP";
  broadcast_msg(confs, cam_count, stop_msg, strlen(stop_msg));

  for (int i = 0; i < cam_count; i++) {
    pthread_join(threads[i], NULL);
  }

  munmap(frameset_buf, shm_size);
  close(shm_fd);
  shm_unlink(SHM_NAME);
  sem_close(consumer_ready);
  sem_close(stop_streams);
  sem_unlink(SEM_CONSUMER_READY);
  sem_unlink(SEM_STOP_STREAMS);
  free(q_bufs);
  free(frame_bufs);
  cleanup_logging();
  return ret;
}
