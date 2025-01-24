const std = @import("std");

pub fn SharedMemBuilder() type {
    return struct {
        const Self = @This();

        buffer: []align(std.mem.page_size) u8,

        const Reservation = struct {
            offset: u64,
            shm_size: u64,
        };

        pub fn reserve(comptime T: type, comptime count: u64, comptime current_size: u64) Reservation {
            const offset = align_up(current_size, @alignOf(T));
            const new_size = current_size + offset + @sizeOf(T) * count;
            return Reservation{
                .offset = offset,
                .shm_size = new_size,
            };
        }

        fn align_up(comptime offset: u64, comptime alignment: u64) u64 {
            return (offset + (alignment - 1)) & ~(alignment - 1);
        }

        pub fn init(shm_name: [*:0]const u8, shm_addr: ?[*]align(std.mem.page_size) u8, size: u64) !Self {
            const flags: std.posix.O = .{
                .ACCMODE = .RDWR,
                .CREAT = true,
            };
            const permissions: std.posix.mode_t = 0o666;
            const fd = std.c.shm_open(shm_name, @bitCast(flags), permissions);
            if (fd == -1) {
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
            defer std.posix.close(fd);

            try std.posix.ftruncate(fd, size);

            const prot_flags: u32 = std.posix.PROT.READ | std.posix.PROT.WRITE;
            const buffer = try std.posix.mmap(shm_addr, @intCast(size), prot_flags, .{ .TYPE = .SHARED }, fd, 0);

            if (shm_addr) |addr| {
                try std.testing.expect(buffer.ptr == addr);
            }

            return .{
                .buffer = buffer,
            };
        }

        pub fn deinit(self: *Self) void {
            std.posix.munmap(self.buffer);
        }

        pub fn get_slice(self: *const Self, T: type, offset: u64, count: u64) []T {
            const typed_ptr: [*]T = @alignCast(@ptrCast(&self.buffer[offset]));
            return typed_ptr[0..count];
        }

        pub fn get_item(self: *const Self, T: type, offset: u64) *T {
            return @alignCast(@ptrCast(&self.buffer[offset]));
        }
    };
}
