#include <stdexcept>

#include "sem_init.hpp"
#include "logging.h"

std::unique_ptr<sem_t, sem_deleter> init_semaphore() {
  sem_t* sem = (sem_t*)malloc(sizeof(sem_t));
  if (!sem) {
    const char* err = "Failed to allocate semaphore memory";
    log_write(ERROR, err);
    throw std::runtime_error(err);
  }

  if (sem_init(sem, 0, 0) < 0) {
    free(sem);
    const char* err = "Failed to initialize semaphore";
    log_write(ERROR, err);
    throw std::runtime_error(err);
  }

  return std::unique_ptr<sem_t, sem_deleter>(sem);
}
