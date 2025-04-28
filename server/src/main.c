#define _GNU_SOURCE
#include <cuda_runtime.h>
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
#define DEV_PTRS_PER_THREAD 16

static void shutdown_handler(int signum);
static void perform_cleanup();

struct cleanup_ctx {
  void* mmap_buf;
  size_t shm_size;
  int shm_fd;
  struct ts_dev_ptr* ts_dev_ptrs;
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

  // shm_block contains everything with a size known at compile time
  struct shm_block_t {
    struct producer_q ipc_handles_pq;
    struct consumer_q ipc_handles_cq;
    void* ipc_handles_q_buf[DEV_PTRS_PER_THREAD];
  };

  size_t shm_size = sizeof(struct shm_block_t);

  // atomic counters for dev ptrs and ipc handle subarrays in use
  shm_size += align_up(shm_size, _Alignof(_Atomic uint32_t));
  const size_t counters_offset = shm_size;
  shm_size += sizeof(_Atomic uint32_t) * (cam_count + 1);

  // cuda ipc mem handles array
  size_t num_ipc_handles = cam_count * DEV_PTRS_PER_THREAD;
  shm_size += align_up(shm_size, _Alignof(cudaIpcMemHandle_t*));
  const size_t ipc_handles_offset = shm_size;
  shm_size += sizeof(cudaIpcMemHandle_t) * (num_ipc_handles);

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

  struct shm_block_t* shm_block = (struct shm_block_t*)(mmap_buf);
  struct producer_q* ipc_handles_pq = &shm_block->ipc_handles_pq;
  struct consumer_q* ipc_handles_cq = &shm_block->ipc_handles_cq;
  void** ipc_handles_q_buf = shm_block->ipc_handles_q_buf;
  spsc_queue_init(
    ipc_handles_pq,
    ipc_handles_cq,
    ipc_handles_q_buf,
    DEV_PTRS_PER_THREAD
  );

  _Atomic uint32_t* counters = (_Atomic uint32_t*)(mmap_buf + counters_offset);

  uint32_t ipc_handles_idx = 0; // increments by cam_count until it hits num_ipc_handles
  cudaIpcMemHandle_t* ipc_handles = (cudaIpcMemHandle_t*)(mmap_buf + ipc_handles_offset);

  struct producer_q dev_ptr_pqs[cam_count];
  struct consumer_q dev_ptr_cqs[cam_count];
  void* dev_ptr_q_bufs[cam_count * DEV_PTRS_PER_THREAD];

  for (int i = 0; i < cam_count; i++) {
    spsc_queue_init(
      &dev_ptr_pqs[i],
      &dev_ptr_cqs[i],
      &dev_ptr_q_bufs[i * DEV_PTRS_PER_THREAD],
      DEV_PTRS_PER_THREAD
    );
  }

  struct ts_dev_ptr* ts_dev_ptrs = malloc(cam_count * DEV_PTRS_PER_THREAD * sizeof(struct ts_dev_ptr));
  if (ts_dev_ptrs == NULL) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Failed to allocate memory for timestamped device ptrs"
    );
    log(ERROR, logstr);
    perform_cleanup();
  }
  cleanup.ts_dev_ptrs = ts_dev_ptrs;

  struct thread_ctx ctxs[cam_count];
  pthread_t threads[cam_count];
  cleanup.threads = threads;
  for (int i = 0; i < cam_count; i++) {
    ctxs[i].conf = &confs[i];
    ctxs[i].stream_conf = &stream_conf;
    ctxs[i].dev_ptr_queue = &dev_ptr_pqs[i];
    ctxs[i].dev_ptrs_used = &counters[i];
    ctxs[i].dev_ptrs_total = DEV_PTRS_PER_THREAD;
    ctxs[i].dev_ptrs = &ts_dev_ptrs[i * DEV_PTRS_PER_THREAD];
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

  struct ts_dev_ptr* dev_ptrs_set[cam_count];
  memset(dev_ptrs_set, 0, sizeof(void*) * cam_count);

  // reuse ts for our sleep timer to prevent busy waiting
  ts.tv_sec = 0;
  ts.tv_nsec = EMPTY_QS_WAIT;

  while (running) {
    // dequeue a full set of timestamped frame buffers from each worker thread
    bool full_set = true;
    for(int i = 0; i < cam_count; i++) {
      if (dev_ptrs_set[i] != NULL)
        continue; // already set

      dev_ptrs_set[i] = spsc_dequeue(&dev_ptr_cqs[i]);

      if (dev_ptrs_set[i] == NULL) {
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
      if (dev_ptrs_set[i]->timestamp > max_timestamp)
        max_timestamp = dev_ptrs_set[i]->timestamp;
    }

    bool all_equal = true;
    for (int i = 0; i < cam_count; i++) {
      if (dev_ptrs_set[i]->timestamp != max_timestamp) {
        all_equal = false;
        atomic_fetch_sub(&counters[i], 1);
        dev_ptrs_set[i] = NULL; // get a new timestamped buffer
      }
    }

    if (!all_equal)
      continue;

    log(BENCHMARK, "Received full frameset");

    // wait for an available ipc_handles subarray
    while (atomic_load_explicit(&counters[cam_count], memory_order_relaxed) >= DEV_PTRS_PER_THREAD) {
      nanosleep(&ts, NULL);
    }
    atomic_fetch_add(&counters[cam_count], 1);

    for (int i = 0; i < cam_count; i++) {
      cudaIpcGetMemHandle(&ipc_handles[ipc_handles_idx + i], dev_ptrs_set[i]->dev_ptr);
    }
    spsc_enqueue(ipc_handles_pq, &ipc_handles[ipc_handles_idx]);
    ipc_handles_idx += cam_count;
    if (ipc_handles_idx >= num_ipc_handles) {
      ipc_handles_idx = 0;
    }
    memset(dev_ptrs_set, 0, sizeof(void*) * cam_count);
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

  if (cleanup.ts_dev_ptrs != NULL)
    free(cleanup.ts_dev_ptrs);

  if (cleanup.logging_initialized)
    cleanup_logging();

  if (cleanup.mmap_buf)
    munmap(cleanup.mmap_buf, cleanup.shm_size);

  if (cleanup.shm_fd >= 0) {
    close(cleanup.shm_fd);
    shm_unlink(SHM_NAME);
  }
}
