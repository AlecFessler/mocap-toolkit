#ifndef TCP_THREAD_H
#define TCP_THREAD_H

#include <queue>
#include <mutex>
#include <pthread.h>

struct shared_data {
  std::queue<void*> queue;
  pthread_mutex_t mutex;
};

void* tcp_thread(void* shared_data);

#endif // TCP_THREAD_H
