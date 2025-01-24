//! Timestamped frame buffer struct
//! The buffer size is required to be comptime known
//! so that other types, like SharedMemBuilder and
//! RingAllocator, are able to determine the full
//! struct size at comptime

pub fn Frame(comptime size: u64) type {
    return extern struct {
        timestamp: u64,
        buffer: [size]u8,
    };
}
