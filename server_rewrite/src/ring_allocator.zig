//! Simple ring buffer for providing a backing memory for an array of items
//! The allocator *DOES NOT* track whether memory is free for usage or not
//! It's intended to be used with a single producer single consumer queue,
//! where the allocator has 2 more slots than the queue itself, meaning the
//! consumer can be operating on an item, the queue can be full, and the
//! producer can safetly call next() on this struct to get the final slot
//! available. The idea is, if the conumser is the queue being full is what
//! controls access to the memory provided by this struct

const std = @import("std");

pub fn RingAllocator(comptime T: type) type {
    return struct {
        const Self = @This();

        buffer: []T,
        index: u64,

        pub fn init(buffer: []T) Self {
            return .{
                .buffer = buffer,
                .index = 0,
            };
        }

        /// Control over safe memory usage is not handled internal to this struct
        /// See the top-level comment for more info
        pub fn next(self: *Self) *T {
            const slot = &self.buffer[self.index];
            self.index += 1;
            if (self.index == self.buffer.len) self.index = 0;
            return slot;
        }
    };
}
