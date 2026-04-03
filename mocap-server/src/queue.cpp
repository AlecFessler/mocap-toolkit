#include <cerrno>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#include "queue.hpp"
#include "logging.h"

int init_queue(
  queue* q,
  uint32_t type_size,
  uint32_t initial_capacity
) {
  q->data = malloc(type_size * initial_capacity);
  if (!q->data) {
    log_write(ERROR, "Failed to allocate buffer for queue");
    return -ENOMEM;
  }

  q->type_size = type_size;
  q->capacity = initial_capacity;
  q->size = 0;
  q->head = 0;
  q->tail = 0;

  return 0;
}

static int resize(queue* q) {
  void* data = malloc(q->type_size * q->capacity * 2);
  if (!data) {
    log_write(ERROR, "Failed to allocate buffer for queue resize");
    return -ENOMEM;
  }

  uint8_t* src = static_cast<uint8_t*>(q->data);
  uint8_t* dst = static_cast<uint8_t*>(data);

  // head at start, full but no wrapping
  if (q->head == 0) {
    memcpy(
      dst,
      src,
      q->type_size * q->capacity
    );
  } else {
    // copy from head to end
    memcpy(
      dst,
      src + (q->head * q->type_size),
      q->type_size * (q->capacity - q->head)
    );
    // copy from start to tail
    memcpy(
      dst + (q->type_size * (q->capacity - q->head)),
      src,
      q->type_size * q->tail
    );
  }

  free(q->data);

  q->data = data;
  q->capacity = q->capacity * 2;
  q->tail = q->size;
  q->head = 0;

  return 0;
}

int enqueue(queue* q, void* data) {
  if (q->size == q->capacity) {
    int ret = resize(q);
    if (ret)
      return ret;
  }

  uint8_t* dst = static_cast<uint8_t*>(q->data);
  memcpy(
    dst + (q->tail * q->type_size),
    data,
    q->type_size
  );

  q->tail = (q->tail + 1) % q->capacity;
  q->size++;

  return 0;
}

int dequeue(queue* q, void* buf) {
  if (q->size == 0)
    return -EAGAIN;

  uint8_t* src = static_cast<uint8_t*>(q->data);
  memcpy(
    buf,
    src + (q->head * q->type_size),
    q->type_size
  );

  q->head = (q->head + 1) % q->capacity;
  q->size--;

  return 0;
}

void cleanup_queue(queue* q) {
  if (q->data)
    free(q->data);
}
