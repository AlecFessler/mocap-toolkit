#ifndef QUEUE_H
#define QUEUE_H

#include <stdint.h>

typedef struct queue {
  void* data;
  uint32_t type_size;
  uint32_t capacity;
  uint32_t size;
  uint32_t head;
  uint32_t tail;
} queue;

int init_queue(
  queue* q,
  uint32_t type_size,
  uint32_t initial_capacity
);
int enqueue(queue* q, void* data);
int dequeue(queue* q, void* buf);
void cleanup_queue(queue* q);

#endif // QUEUE_H
