#include "tcp_thread.h"
#include <memory>
#include <sched.h>
#include <signal.h>
#include <sys/prctl.h>
#include "Logger.h"

void* tcp_thread(void* shared_data_ptr) {
  prctl(PR_SET_PDEATHSIG, SIGTERM);

  std::unique_ptr<Logger> logger;
  try {
    logger = std::make_unique<Logger>("tcp_logs.txt");
  } catch (const std::exception& e) {
    return nullptr;
  }

  struct sched_param param;
  param.sched_priority = sched_get_priority_max(SCHED_FIFO) - 1;
  if (sched_setscheduler(0, SCHED_FIFO, &param) < 0) {
    const char* err = "Failed to set real-time scheduling policy";
    logger->log(Logger::Level::ERROR, __FILE__, __LINE__, err);
    return nullptr;
  }

  shared_data* s_data = static_cast<shared_data*>(shared_data_ptr);
  std::queue<void*>& queue = s_data->queue;
  pthread_mutex_t& mutex = s_data->mutex;

  while (true) {
    pthread_mutex_lock(&mutex);
    if (!queue.empty()) {
      void* image = queue.front();
      queue.pop();
      static const char* info = "Received image from queue";
      logger->log(Logger::Level::INFO, __FILE__, __LINE__, info);
    }
    pthread_mutex_unlock(&mutex);
  }

  return nullptr;
}
