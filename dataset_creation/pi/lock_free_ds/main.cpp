#include "lock_free_queue.h"
#include <iostream>
#include <pthread.h>
#include <unistd.h>

void* producer(void* arg) {
    lock_free_queue_t* queue = static_cast<lock_free_queue_t*>(arg);

    for (int i = 0; i < 10; ++i) {
        int* data = new int(i);  // dynamically allocate memory to simulate real data
        if (queue->enqueue(static_cast<void*>(data))) {
            std::cout << "Produced: " << *data << std::endl;
        } else {
            std::cout << "Queue is full, failed to produce: " << *data << std::endl;
            delete data;  // clean up if enqueue failed
        }
        sleep(1);  // Simulate work being done by sleeping for 1 second
    }
    return nullptr;
}

void* consumer(void* arg) {
    lock_free_queue_t* queue = static_cast<lock_free_queue_t*>(arg);

    for (int i = 0; i < 10; ++i) {
        void* data = queue->dequeue();
        if (data != nullptr) {
            std::cout << "Consumed: " << *static_cast<int*>(data) << std::endl;
            delete static_cast<int*>(data);  // clean up the memory after consumption
        } else {
            std::cout << "Queue is empty, failed to consume" << std::endl;
        }
        sleep(2);  // Simulate work being done by sleeping for 2 seconds
    }
    return nullptr;
}

int main() {
    lock_free_queue_t queue(2, 32);  // Initialize queue with 2 thread and preallocated size of 32

    pthread_t producer_thread, consumer_thread;

    pthread_create(&producer_thread, nullptr, producer, &queue);
    pthread_create(&consumer_thread, nullptr, consumer, &queue);

    pthread_join(producer_thread, nullptr);
    pthread_join(consumer_thread, nullptr);

    return 0;
}
