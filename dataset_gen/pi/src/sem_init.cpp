#include "sem_init.h"
#include "logger.h"
#include <stdexcept>

extern std::unique_ptr<logger_t> logger;

std::unique_ptr<sem_t, sem_deleter> init_semaphore() {
  sem_t* sem = (sem_t*)malloc(sizeof(sem_t));
  if (!sem) {
    const char* err = "Failed to allocate semaphore memory";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  if (sem_init(sem, 0, 0) < 0) {
    free(sem);
    const char* err = "Failed to initialize semaphore";
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, err);
    throw std::runtime_error(err);
  }

  return std::unique_ptr<sem_t, sem_deleter>(sem);
}
