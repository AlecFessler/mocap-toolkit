#ifndef TCP_THREAD_H
#define TCP_THREAD_H

#include <pthread.h>
#include <queue>
#include <memory>
#include <mutex>
#include "shared_defs.h"

struct shared_data {
  std::queue<std::unique_ptr<unsigned char[]>> queue;
  pthread_mutex_t mutex;
};

void* tcp_thread(void* shared_data);

#endif // TCP_THREAD_H
