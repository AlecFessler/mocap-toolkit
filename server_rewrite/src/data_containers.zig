pub fn Packet(comptime size: u64) type {
    return struct {
        timestamp: u64,
        buffer: [size]u8,
    };
}

/// Must be extern struct for guaranteed memory layout in shared memory
pub fn Frame(comptime size: u64) type {
    return extern struct {
        timestamp: u64,
        buffer: [size]u8,
    };
}

pub fn Frameset(comptime size: u64) type {
    return struct {
        buffer: [size]*Frame,
    };
}
