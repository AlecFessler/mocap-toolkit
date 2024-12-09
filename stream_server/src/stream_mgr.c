#define _GNU_SOURCE // for cpuset macros
#include <errno.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "stream_mgr.h"
#include "logging.h"

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

void* stream_mgr(void* ptr) {
  struct thread_ctx* ctx = (struct thread_ctx*)ptr;

  pin_to_core(ctx->core);

  return NULL;
}
