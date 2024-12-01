#ifndef SEM_INIT_H
#define SEM_INIT_H

#include <cstdlib>
#include <memory>
#include <semaphore.h>

struct sem_deleter {
  void operator()(sem_t* sem) const {
    if (sem) {
      sem_destroy(sem);
      free(sem);
    }
  }
};

std::unique_ptr<sem_t, sem_deleter> init_semaphore();

#endif // SEM_INIT_H
