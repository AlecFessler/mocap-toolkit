#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <semaphore.h>
#include <sched.h>
#include <signal.h>
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

#define CORES_PER_CCD 8
#define TIMESTAMP_DELAY 1 // seconds
#define EMPTY_QS_WAIT 10000 // 0.01 ms
#define FRAME_BUFS_PER_THREAD 256

static void shutdown_handler(int signum);
static void perform_cleanup();

struct cleanup_ctx {
  void* frame_bufs;
  void* q_bufs;
  void* frameset_buf;
  size_t shm_size;
  int shm_fd;
  sem_t* consumer_ready;
  pthread_t* threads;
  int thread_count;
  bool logging_initialized;
};

static struct cleanup_ctx cleanup = {
  0,
  .shm_fd = -1
};

static volatile sig_atomic_t running = 1;

int main() {
  int ret = 0;
  char logstr[128];

  ret = setup_logging(LOG_PATH);
  if (ret) {
    printf("Error opening log file: %s\n", strerror(errno));
    return -errno;
  }
  cleanup.logging_initialized = true;

  struct sigaction sa = {
    .sa_handler = shutdown_handler,
    .sa_flags = 0
  };
  sigemptyset(&sa.sa_mask);
  ret = sigaction(SIGTERM, &sa, NULL);
  if (ret == -1) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Failed to set signal handler: %s",
      strerror(errno)
    );
    log(ERROR, logstr);
  }
  ret = sigaction(SIGINT, &sa, NULL);
  if (ret == -1) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Failed to set signal handler: %s",
      strerror(errno)
    );
    log(ERROR, logstr);
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
    perform_cleanup();
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
    perform_cleanup();
    return ret;
  }

  // pin to cam_count % 8 to stay on ccd0 for 3dv cache with threads
  // but not be on the same core as any threads until there are 8+
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cam_count % CORES_PER_CCD, &cpuset);
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
    perform_cleanup();
    return -errno;
  }

  const uint64_t frame_bufs_count = cam_count * FRAME_BUFS_PER_THREAD;
  const uint64_t frame_buf_size = DECODED_FRAME_WIDTH * DECODED_FRAME_HEIGHT * 3 / 2;

  uint8_t* frame_bufs = malloc(frame_bufs_count * frame_buf_size);
  if (!frame_bufs) {
    log(ERROR, "Failed to allocate frame buffers");
    perform_cleanup();
    return -ENOMEM;
  }
  cleanup.frame_bufs = frame_bufs;

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
    perform_cleanup();
    return -ENOMEM;
  }
  cleanup.q_bufs = q_bufs;

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
  cleanup.threads = threads;
  for (int i = 0; i < cam_count; i++) {
    ctxs[i].conf = &confs[i];
    ctxs[i].filled_bufs = &filled_frame_producer_qs[i];
    ctxs[i].empty_bufs = &empty_frame_consumer_qs[i];
    ctxs[i].core = i % CORES_PER_CCD;
    ctxs[i].main_thread = pid;

    ret = pthread_create(
      &threads[i],
      NULL,
      stream_mgr_fn,
      (void*)&ctxs[i]
    );

    if (ret) {
      log(ERROR, "Error spawning thread");
      perform_cleanup();
      return ret;
    }

    cleanup.thread_count++;
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
    perform_cleanup();
    return -errno;
  }
  cleanup.consumer_ready = consumer_ready;

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
    perform_cleanup();
  }
  cleanup.shm_fd = shm_fd;

  size_t shm_size = frame_buf_size * cam_count + sizeof(uint64_t);
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
    perform_cleanup();
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
    perform_cleanup();
  }
  cleanup.frameset_buf = frameset_buf;

  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  uint64_t timestamp = (ts.tv_sec + TIMESTAMP_DELAY) * 1000000000ULL + ts.tv_nsec;
  broadcast_msg(confs, cam_count, (char*)&timestamp, sizeof(timestamp));

  struct ts_frame_buf* current_frames[cam_count];
  memset(current_frames, 0, sizeof(struct ts_frame_buf*) * cam_count);

  // reuse ts for our sleep timer to prevent busy waiting
  ts.tv_sec = 0;
  ts.tv_nsec = EMPTY_QS_WAIT;

  while (running) {
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

    if (!full_set) { // need a full set to proceed
      nanosleep(&ts, NULL);
      continue;
    }

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
      "Received frameset with timestamp %lu",
      max_timestamp
    );
    log(DEBUG, logstr);

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
      uint64_t* timestamp_ptr = frameset_buf + (frame_buf_size * cam_count);
      *timestamp_ptr = max_timestamp;
      sem_post(consumer_ready);
    }

    // get a new full set
    for (int i = 0; i < cam_count; i++) {
      spsc_enqueue(&empty_frame_producer_qs[i], current_frames[i]);
      current_frames[i] = NULL;
    }
  }

  // stop the camera devices
  const char* stop_msg = "STOP";
  broadcast_msg(confs, cam_count, stop_msg, strlen(stop_msg));

  perform_cleanup();
  return ret;
}

static void shutdown_handler(int signum) {
  (void)signum;
  running = 0;
}

static void perform_cleanup() {
  if (cleanup.frameset_buf)
    munmap(cleanup.frameset_buf, cleanup.shm_size);

  if (cleanup.shm_fd >= 0) {
    close(cleanup.shm_fd);
    shm_unlink(SHM_NAME);
  }

  if (cleanup.consumer_ready) {
    sem_close(cleanup.consumer_ready);
    sem_unlink(SEM_CONSUMER_READY);
  }

  if (cleanup.threads) {
    for (int i = 0; i < cleanup.thread_count; i++) {
      pthread_kill(cleanup.threads[i], SIGUSR2);
    }

    for (int i = 0; i < cleanup.thread_count; i++) {
      pthread_join(cleanup.threads[i], NULL);
    }
  }

  if (cleanup.q_bufs)
    free(cleanup.q_bufs);

  if (cleanup.frame_bufs)
    free(cleanup.frame_bufs);

  if (cleanup.logging_initialized)
    cleanup_logging();
}
