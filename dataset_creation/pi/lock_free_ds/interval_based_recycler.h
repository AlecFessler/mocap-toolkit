#ifndef INTERVAL_BASED_RECYCLER_H
#define INTERVAL_BASED_RECYCLER_H

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include "lock_free_stack.h"

struct reservation_t {
    std::uint64_t lower;
    std::uint64_t upper;
};

template <typename T>
struct memory_block_t {
    std::uint64_t created_epoch;
    std::uint64_t retired_epoch;
    T* data;
};

template <typename T>
class interval_based_recycler_t {
public:
    interval_based_recycler_t(int num_threads, int prealloc_size, int epoch_freq, int recycle_freq, int recycle_batch_size);
    ~interval_based_recycler_t();

    int thread_idx();
    memory_block_t<T>* get_block();
    void retire_block(memory_block_t<T>* block);
    void start_op();
    void end_op();
    T* read_block(memory_block_t<T>* block);

private:
    std::atomic<std::uint64_t> epoch;
    int num_threads_;
    reservation_t* reservations;

    char* mem_block;

    int epoch_freq_;
    int recycle_freq_;
    int recycle_batch_size_;
    std::atomic<std::uint64_t> thread_idx_counter;
    static thread_local std::uint64_t counter;

    lock_free_stack_t available_blocks;
    lock_free_stack_t empty_stack_nodes;
    lock_free_stack_t retired_blocks;

    void recycle_blocks();
};

template <typename T>
thread_local std::uint64_t interval_based_recycler_t<T>::counter = 0;

template <typename T>
interval_based_recycler_t<T>::interval_based_recycler_t(int num_threads,
                                                        int prealloc_size,
                                                        int epoch_freq,
                                                        int recycle_freq,
                                                        int recycle_batch_size) {
    epoch.store(0);

    num_threads_ = num_threads;
    reservations = new reservation_t[num_threads];
    for (int i = 0; i < num_threads; i++)
        reservations[i].lower = reservations[i].upper = std::numeric_limits<std::uint64_t>::max();

    epoch_freq_ = epoch_freq;
    recycle_freq_ = recycle_freq;
    recycle_batch_size_ = recycle_batch_size;

    mem_block = new alignas(std::max({alignof(stack_node_t),
                                      alignof(memory_block_t<T>),
                                      alignof(T)}))
                char[prealloc_size * sizeof(stack_node_t) +
                     prealloc_size * sizeof(memory_block_t<T>) +
                     prealloc_size * sizeof(T)];

    for (int i = 0; i < prealloc_size; i++) {
        int stack_node_offset = i * sizeof(stack_node_t);
        int memory_block_offset = stack_node_offset + sizeof(stack_node_t);
        int container_node_offset = memory_block_offset + sizeof(memory_block_t<T>);

        stack_node_t* stack_node = (stack_node_t*)(mem_block + stack_node_offset);
        memory_block_t<T>* memory_block = (memory_block_t<T>*)(mem_block + memory_block_offset);
        T* container_node = (T*)(mem_block + container_node_offset);

        stack_node->ptr = (void*)memory_block;
        memory_block->data = (T*)container_node;

        available_blocks.push(stack_node);
    }
}

template <typename T>
interval_based_recycler_t<T>::~interval_based_recycler_t() {
    delete[] reservations;
    delete[] mem_block;
}

template <typename T>
int interval_based_recycler_t<T>::thread_idx() {
    // using more than num_threads_ at once is not supported
    // and is undefined behavior
    static thread_local int thread_idx = thread_idx_counter.fetch_add(1, std::memory_order_acq_rel) % num_threads_;
    return thread_idx;
}

template <typename T>
memory_block_t<T>* interval_based_recycler_t<T>::get_block() {
    if (available_blocks.empty())
        recycle_blocks();

    if (++counter % epoch_freq_ == 0) // thread local counter
        epoch.fetch_add(1, std::memory_order_relaxed);

    stack_node_t* node = available_blocks.pop();
    if (!node)
        // this should not happen unless,
        // there are not enough allocated blocks or,
        // this thread is under high contention
        return nullptr;

    memory_block_t<T>* block = (memory_block_t<T>*)node->ptr;
    node->ptr = nullptr;
    empty_stack_nodes.push(node);

    std::uint64_t current_epoch = epoch.load(std::memory_order_acquire);
    block->created_epoch = current_epoch;

    return block;
}

template <typename T>
void interval_based_recycler_t<T>::retire_block(memory_block_t<T>* block) {
    if (++counter % recycle_freq_ == 0) // thread local counter
        recycle_blocks();

    stack_node_t* node = empty_stack_nodes.pop();
    if (!node)
        // this should not ever happen
        // there are 1:1 empty stack nodes for retireable blocks
        return;
    node->ptr = (void*)block;

    std::uint64_t current_epoch = epoch.load(std::memory_order_acquire);
    block->retired_epoch = current_epoch;

    retired_blocks.push(node);
}

template <typename T>
void interval_based_recycler_t<T>::start_op() {
    int tidx = thread_idx();
    std::uint64_t current_epoch = epoch.load(std::memory_order_acquire);
    reservations[tidx].lower = current_epoch;
    reservations[tidx].upper = current_epoch;
}

template <typename T>
void interval_based_recycler_t<T>::end_op() {
    int tidx = thread_idx();
    std::uint64_t current_epoch = epoch.load(std::memory_order_acquire);
    reservations[tidx].lower = std::numeric_limits<std::uint64_t>::max();
    reservations[tidx].upper = std::numeric_limits<std::uint64_t>::max();
}

template <typename T>
T* interval_based_recycler_t<T>::read_block(memory_block_t<T>* block) {
    int tidx = thread_idx();
    reservations[tidx].upper = block->created_epoch;
    T* data = (T*)block->data;
    return data;
}

template <typename T>
void interval_based_recycler_t<T>::recycle_blocks() {
    int reserved_idx = 0;
    stack_node_t* reserved_blocks[recycle_batch_size_];

    while (!retired_blocks.empty() && reserved_idx < recycle_batch_size_) {
        stack_node_t* node = retired_blocks.pop();
        if (!node)
            break;
        memory_block_t<T>* block = (memory_block_t<T>*)node->ptr;

        bool reserved = false;
        for (int i = 0; i < num_threads_; i++) {
            if (block->created_epoch <= reservations[i].upper && block->retired_epoch >= reservations[i].lower) {
                reserved = true;
                break;
            }
        }

        if (reserved)
            reserved_blocks[reserved_idx++] = node;
        else
            available_blocks.push(node);
    }

    for (int i = 0; i < reserved_idx; i++)
        retired_blocks.push(reserved_blocks[i]);
}

#endif // INTERVAL_BASED_RECYCLER_H
