#include "tcp_thread.h"
#include "logger.h"
#include <memory>
#include <sched.h>
#include <signal.h>
#include <sys/prctl.h>

void* tcp_thread(void* shared_data_ptr) {
  /*********************/
  /* Initialize logger */
  /*********************/

  std::unique_ptr<logger_t> logger;
  try {
    logger = std::make_unique<logger_t>("tcp_logs.txt");
  } catch (const std::exception& e) {
    return nullptr;
  }

  /*****************************************************************************/
  /* Set real-time scheduling policy to one less than max (and parent process) */
  /*****************************************************************************/

  struct sched_param param;
  param.sched_priority = sched_get_priority_max(SCHED_FIFO) - 1;
  if (sched_setscheduler(0, SCHED_FIFO, &param) < 0) {
    const char* err = "Failed to set real-time scheduling policy";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    return nullptr;
  }

  /*************************************/
  /* Cast shared data and grab handles */
  /*************************************/

  shared_data* s_data = static_cast<shared_data*>(shared_data_ptr);
  volatile sig_atomic_t& running = s_data->running;
  std::queue<std::unique_ptr<unsigned char[]>>& queue = s_data->queue;
  pthread_mutex_t& mutex = s_data->mutex;
  pthread_cond_t& cond = s_data->cond;

  /***********************************************************/
  /* Poll queue for images, and send over tcp when available */
  /***********************************************************/

  while (running) {
    pthread_mutex_lock(&mutex);
    pthread_cond_wait(&cond, &mutex);

    unsigned char* image = queue.front().get();
    queue.pop();

    static const char* info = "Received image from queue";
    logger->log(logger_t::level_t::INFO, __FILE__, __LINE__, info);

    pthread_mutex_unlock(&mutex);
  }

  return nullptr;
}
