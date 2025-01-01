#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <spsc_queue.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "logging.h"
#include "parse_conf.h"
#include "stream_mgr.h"
#include "network.h"

#define LOG_PATH "/var/log/mocap-toolkit/server.log"
#define CAM_CONF_PATH "/etc/mocap-toolkit/cams.yaml"

#define SHM_NAME "/mocap-toolkit_shm"
#define SHM_ADDR ((void*)0x7f0000000000)

#define CORES_PER_CCD 8
#define TIMESTAMP_DELAY 1 // seconds
#define EMPTY_QS_WAIT 10000 // 0.01 ms
#define FRAME_BUFS_PER_THREAD 512
#define FRAMESET_SLOTS_PER_THREAD 8

static void shutdown_handler(int signum);
static void perform_cleanup();

struct cleanup_ctx {
  void* mmap_buf;
  size_t shm_size;
  int shm_fd;
  pthread_t* threads;
  int thread_count;
  bool logging_initialized;
};

static struct cleanup_ctx cleanup = {
  0,
  .shm_fd = -1
};

static volatile sig_atomic_t running = 1;

int main(int argc, char* argv[]) {
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

  struct stream_conf stream_conf;
  struct cam_conf confs[cam_count];
  ret = parse_conf(&stream_conf, confs, cam_count);
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

  // filter out target cam, otherwise stream with all cams in config
  int target_cam_id;
  if (argc > 1) {
    target_cam_id = atoi(argv[1]);

    bool found = false;
    for (int i = 0; i < cam_count; i++) {
      if (confs[i].id != target_cam_id)
        continue;

      confs[0] = confs[i];
      cam_count = 1;
      found = true;
      break;
    }

    if (!found) {
      snprintf(
        logstr,
        sizeof(logstr),
        "Camera ID %d not found in config",
        target_cam_id
      );
      log(ERROR, logstr);
      cleanup_logging();
      return -EINVAL;
    }
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

  #define align_up(offset, align) (((offset) + (align-1)) & ~(align-1))

  // frame buffers
  const uint64_t frame_bufs_count = cam_count * FRAME_BUFS_PER_THREAD;
  const uint64_t frame_buf_size = stream_conf.frame_width * stream_conf.frame_height * 3 / 2;
  size_t shm_size = frame_buf_size * frame_bufs_count;

  // timestamped structs with frame buffer ptrs
  shm_size = align_up(shm_size, _Alignof(struct ts_frame_buf));
  size_t ts_frame_bufs_offset = shm_size;
  shm_size += sizeof(struct ts_frame_buf) * frame_bufs_count;

  // filled frameset producer queue
  shm_size = align_up(shm_size, _Alignof(struct producer_q));
  size_t filled_frameset_producer_q_offset = shm_size;
  shm_size += sizeof(struct producer_q);

  // filled frameset consumer queue
  shm_size = align_up(shm_size, _Alignof(struct consumer_q));
  size_t filled_frameset_consumer_q_offset = shm_size;
  shm_size += sizeof(struct consumer_q);

  // empty frameset producer queue
  shm_size = align_up(shm_size, _Alignof(struct producer_q));
  size_t empty_frameset_producer_q_offset = shm_size;
  shm_size += sizeof(struct producer_q);

  // empty frameset consumer queue
  shm_size = align_up(shm_size, _Alignof(struct consumer_q));
  size_t empty_frameset_consumer_q_offset = shm_size;
  shm_size += sizeof(struct consumer_q);

  // filled frameset queue buffer
  shm_size = align_up(shm_size, _Alignof(void**));
  size_t filled_q_bufs_offset = shm_size;
  shm_size += (sizeof(void*) * FRAME_BUFS_PER_THREAD);

  // empty frameset queue buffer
  shm_size = align_up(shm_size, _Alignof(void**));
  size_t empty_q_bufs_offset = shm_size;
  shm_size += (sizeof(void*) * FRAME_BUFS_PER_THREAD);

  // framesets slots
  size_t num_frameset_slots = FRAMESET_SLOTS_PER_THREAD * cam_count;
  shm_size = align_up(shm_size, _Alignof(struct ts_frame_buf*));
  size_t frameset_slots_offset = shm_size;
  shm_size += sizeof(struct ts_frame_buf*) * num_frameset_slots;

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
  cleanup.shm_size = shm_size;

  uint8_t* mmap_buf = mmap(
    SHM_ADDR,
    shm_size,
    PROT_READ | PROT_WRITE,
    MAP_SHARED,
    shm_fd,
    0
  );
  if (mmap_buf == MAP_FAILED) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error mapping shared memory: %s",
      strerror(errno)
    );
    log(ERROR, logstr);
    perform_cleanup();
  }
  cleanup.mmap_buf = mmap_buf;

  // assign frame buffers to ts_frame_bufs (this is a permanent assignment)
  uint8_t* frame_bufs = mmap_buf;
  struct ts_frame_buf* ts_frame_bufs = (struct ts_frame_buf*)(mmap_buf + ts_frame_bufs_offset);
  for (uint i = 0; i < frame_bufs_count; i++) {
    size_t offset = i * frame_buf_size;
    ts_frame_bufs[i].frame_buf = frame_bufs + offset;
  }

  struct producer_q* filled_frameset_producer_q = (struct producer_q*)(mmap_buf + filled_frameset_producer_q_offset);
  struct consumer_q* filled_frameset_consumer_q = (struct consumer_q*)(mmap_buf + filled_frameset_consumer_q_offset);
  void* filled_frameset_q_bufs = (void**)(mmap_buf + filled_q_bufs_offset);
  spsc_queue_init(
    filled_frameset_producer_q,
    filled_frameset_consumer_q,
    filled_frameset_q_bufs,
    num_frameset_slots
  );

  struct producer_q* empty_frameset_producer_q = (struct producer_q*)(mmap_buf + empty_frameset_producer_q_offset);
  struct consumer_q* empty_frameset_consumer_q = (struct consumer_q*)(mmap_buf + empty_frameset_consumer_q_offset);
  void* empty_frameset_q_bufs = (void**)(mmap_buf + empty_q_bufs_offset);
  spsc_queue_init(
    empty_frameset_producer_q,
    empty_frameset_consumer_q,
    empty_frameset_q_bufs,
    num_frameset_slots
  );

  struct ts_frame_buf** frameset_slots = (struct ts_frame_buf**)(mmap_buf + frameset_slots_offset);
  memset(frameset_slots, 0, sizeof(struct ts_frame_buf*) * num_frameset_slots);
  for (size_t i = 0; i < num_frameset_slots; i++) {
    struct ts_frame_buf** slot_addr = frameset_slots + (i * cam_count);
    spsc_enqueue(
      empty_frameset_producer_q,
      slot_addr
    );
  }

  struct producer_q filled_frame_producer_qs[cam_count];
  struct consumer_q filled_frame_consumer_qs[cam_count];

  struct producer_q empty_frame_producer_qs[cam_count];
  struct consumer_q empty_frame_consumer_qs[cam_count];

  void* q_bufs[frame_bufs_count * 2];
  for (int i = 0; i < cam_count; i++) {
    spsc_queue_init(
      &filled_frame_producer_qs[i],
      &filled_frame_consumer_qs[i],
      &q_bufs[i * 2 * FRAME_BUFS_PER_THREAD],
      FRAME_BUFS_PER_THREAD
    );

    spsc_queue_init(
      &empty_frame_producer_qs[i],
      &empty_frame_consumer_qs[i],
      &q_bufs[(i * 2 + 1) * FRAME_BUFS_PER_THREAD],
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
    ctxs[i].stream_conf = &stream_conf;
    ctxs[i].filled_bufs = &filled_frame_producer_qs[i];
    ctxs[i].empty_bufs = &empty_frame_consumer_qs[i];
    ctxs[i].core = i % CORES_PER_CCD;
    ctxs[i].main_running = &running;

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

  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  uint64_t timestamp = (ts.tv_sec + TIMESTAMP_DELAY) * 1000000000ULL + ts.tv_nsec;
  broadcast_msg(confs, cam_count, (char*)&timestamp, sizeof(timestamp));

  struct ts_frame_buf* current_frames[cam_count];
  memset(current_frames, 0, sizeof(struct ts_frame_buf*) * cam_count);

  // reuse ts for our sleep timer to prevent busy waiting
  ts.tv_sec = 0;
  ts.tv_nsec = EMPTY_QS_WAIT;

  uint32_t dequeue_retry_counter = 0;

  uint32_t dequeued_framesets = 0;

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

    struct ts_frame_buf** frameset = spsc_dequeue(empty_frameset_consumer_q);
    while (!frameset) {
      if (++dequeue_retry_counter >= stream_conf.fps) {
        log(ERROR, "Main thread ran out of empty frameset dequeue retries");
        running = 0;
        break;
      }
      nanosleep(&ts, NULL);
      frameset = spsc_dequeue(empty_frameset_consumer_q);
    }

    dequeue_retry_counter = 0;

    if (dequeued_framesets >= FRAMESET_SLOTS_PER_THREAD) {
      for (int i = 0; i < cam_count; i++) {
        spsc_enqueue(&empty_frame_producer_qs[i], frameset[i]);
      }
    } else {
      dequeued_framesets++;
    }

    memcpy(frameset, current_frames, sizeof(struct ts_frame_buf*) * cam_count);
    spsc_enqueue(filled_frameset_producer_q, frameset);
    memset(current_frames, 0, sizeof(struct ts_frame_buf*) * cam_count);
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
  if (cleanup.threads) {
    for (int i = 0; i < cleanup.thread_count; i++) {
      pthread_kill(cleanup.threads[i], SIGUSR2);
    }

    for (int i = 0; i < cleanup.thread_count; i++) {
      pthread_join(cleanup.threads[i], NULL);
    }
  }

  if (cleanup.logging_initialized)
    cleanup_logging();

  if (cleanup.mmap_buf)
    munmap(cleanup.mmap_buf, cleanup.shm_size);

  if (cleanup.shm_fd >= 0) {
    close(cleanup.shm_fd);
    shm_unlink(SHM_NAME);
  }
}
