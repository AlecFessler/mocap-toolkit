#ifndef TCP_THREAD_H
#define TCP_THREAD_H

#include <pthread.h>
#include <queue>
#include <memory>
#include <mutex>
#include <signal.h>
#include "shared_defs.h"

struct shared_data {
  volatile sig_atomic_t& running;
  std::queue<std::unique_ptr<unsigned char[]>> queue;
  pthread_mutex_t mutex;
  pthread_cond_t cond;

  shared_data(volatile sig_atomic_t& running_ref) : running(running_ref), mutex(PTHREAD_MUTEX_INITIALIZER), cond(PTHREAD_COND_INITIALIZER) {};
};

void* tcp_thread(void* shared_data);

#endif // TCP_THREAD_H
