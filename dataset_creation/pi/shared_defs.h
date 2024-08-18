#ifndef SHARED_DEFS_H
#define SHARED_DEFS_H

#include <semaphore.h>
#include <signal.h>

constexpr size_t IMAGE_WIDTH = 1920;
constexpr size_t IMAGE_HEIGHT = 1080;
constexpr size_t IMAGE_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT;

/**
 * Shared memory layout:
 * 1. sig_atomic_t running_flag - used by both parent and child processes as well as child's thread
 * 2. sem_t parent_process_ready_sem - used by child process to wait for parent process to initialize
 * 3. sem_t child_process_ready_sem - used by parent process to wait for child process to initialize
 * 4. sem_t img_write_sem - used by child process to signal its ready for a new image
 * 5. sem_t img_read_sem - used by parent process to signal its wrote a new image
 * 6. char[IMAGE_BYTES] img_data - written by parent process, read by child process
 */

struct shared_mem_layout {
  alignas(alignof(sig_atomic_t)) sig_atomic_t running;
  alignas(alignof(sem_t)) sem_t parent_process_ready_sem;
  alignas(alignof(sem_t)) sem_t child_process_ready_sem;
  alignas(alignof(sem_t)) sem_t img_write_sem;
  alignas(alignof(sem_t)) sem_t img_read_sem;
  alignas(alignof(char)) unsigned char img_data[IMAGE_BYTES];
};

#endif
