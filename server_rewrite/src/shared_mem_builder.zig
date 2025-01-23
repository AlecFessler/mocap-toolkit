const std = @import("std");

pub const SharedMemBuilder = struct {
    const Self = @This();

    fd: i32,
    buffer: []align(std.mem.page_size) u8,
    size: u64,

    pub fn init() Self {
        return .{
            .fd = -1,
            .buffer = undefined,
            .size = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        std.posix.munmap(self.buffer);
    }

    fn align_up(offset: u64, alignment: u64) u64 {
        return (offset + (alignment - 1)) & ~(alignment - 1);
    }

    pub fn reserve(self: *Self, T: type, count: u64) u64 {
        const offset = align_up(self.size, @alignOf(T));
        self.size += offset + @sizeOf(T) * count;
        return offset;
    }

    pub fn allocate(self: *Self, shm_name: [*:0]const u8, shm_addr: ?[*]align(std.mem.page_size) u8) !void {
        const flags: std.posix.O = .{
            .ACCMODE = .RDWR,
            .CREAT = true,
        };
        const permissions: std.posix.mode_t = 0o666;
        self.fd = std.c.shm_open(shm_name, @bitCast(flags), permissions);
        if (self.fd == -1) {
            const errnum: u32 = @bitCast(std.c._errno().*);
            const err: std.posix.E = @enumFromInt(errnum);
            switch (err) {
                .SUCCESS => unreachable,
                .ACCES => return error.AccessDenied,
                .EXIST => return error.PathAlreadyExists,
                .INVAL => return error.InvalidInput,
                .MFILE => return error.ProcessFdQuotaExceeded,
                .NAMETOOLONG => return error.NameTooLong,
                .NFILE => return error.SystemFdQuotaExceeded,
                .NOENT => return error.FileNotFound,
                else => return std.posix.unexpectedErrno(err),
            }
        }
        defer std.posix.close(self.fd);

        try std.posix.ftruncate(self.fd, self.size);

        const prot_flags: u32 = std.posix.PROT.READ | std.posix.PROT.WRITE;
        self.buffer = try std.posix.mmap(shm_addr, @intCast(self.size), prot_flags, .{ .TYPE = .SHARED }, self.fd, 0);

        if (shm_addr) |addr| {
            try std.testing.expect(self.buffer.ptr == addr);
        }
    }

    pub fn get_slice(self: *Self, T: type, offset: u64, count: u64) []T {
        const typed_ptr: [*]T = @alignCast(@ptrCast(&self.buffer[offset]));
        return typed_ptr[0..count];
    }
};
