#include <stdexcept>

#include "sem_init.h"
#include "logging.h"

std::unique_ptr<sem_t, sem_deleter> init_semaphore() {
  /**
   * Initializes a POSIX semaphore with automatic cleanup using RAII.
   *
   * This function creates a process-private semaphore that plays a crucial role
   * in coordinating between the camera's capture completion signals and the main
   * processing loop. Unlike a simple mutex, this semaphore can track the number
   * of available frames, acting as both a synchronization and counting mechanism.
   *
   * The initialization process involves two steps:
   * 1. Memory allocation: Uses malloc instead of new because POSIX semaphores
   *    require specific memory alignment which the C allocator guarantees.
   *
   * 2. Semaphore initialization: Creates a process-private semaphore (pshared=0)
   *    with an initial value of 0, meaning the main loop will block until the
   *    first frame is captured.
   *
   * The function uses std::unique_ptr with a custom deleter (sem_deleter) to
   * ensure both proper cleanup order and exception safety:
   * - If sem_init fails, the allocated memory is freed
   * - When the unique_ptr goes out of scope, sem_destroy is called before free
   *
   * Returns:
   *   A unique_ptr to the initialized semaphore with custom cleanup
   *
   * Throws:
   *   std::runtime_error: If memory allocation fails
   *   std::runtime_error: If semaphore initialization fails
   */
  sem_t* sem = (sem_t*)malloc(sizeof(sem_t));
  if (!sem) {
    const char* err = "Failed to allocate semaphore memory";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }

  if (sem_init(sem, 0, 0) < 0) {
    free(sem);
    const char* err = "Failed to initialize semaphore";
    LOG(ERROR, err);
    throw std::runtime_error(err);
  }

  return std::unique_ptr<sem_t, sem_deleter>(sem);
}
