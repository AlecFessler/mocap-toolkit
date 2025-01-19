//! A thread and signal safe logger implementation for posix machines

const std = @import("std");

/// Single system wide log file descriptor
var fd_handle: ?std.posix.fd_t = null;

/// Opens the log file in WRITE ONLY and atomic appends mode
/// Creates the file if it doesn't already exist
pub fn setup(path: []const u8) std.posix.OpenError!void {
    const flags: std.posix.O = .{ .ACCMODE = std.posix.ACCMODE.WRONLY, .CREAT = true, .APPEND = true };
    fd_handle = try std.posix.open(path, flags, 0o664);
}

/// Closes the file descriptor
/// No op if the file isn't opened yet
pub fn cleanup() void {
    if (fd_handle) |fd| {
        std.posix.close(fd);
    }
}

/// Creates a formatted UTC timestamp
/// The formatting is signal safe, unlike common string formatting functions from libc
fn timestamp() std.posix.ClockGetTimeError!u64 {
    var ts: std.posix.timespec = .{ .sec = 0, .nsec = 0 };
    try std.posix.clock_gettime(.REALTIME, &ts);
    const seconds: u64 = @intCast(ts.sec);
    const nanos: u64 = @intCast(ts.nsec);
    std.debug.print("Seconds {}, nanos {}", .{ seconds, nanos });
    return seconds;
}

const LogWriteError = std.posix.ClockGetTimeError || std.posix.WriteError;

/// Atomically appends the bytes to the log file
/// No op if the file isn't opened yet
pub fn write(bytes: []const u8) LogWriteError!void {
    if (fd_handle) |fd| {
        _ = try timestamp();
        _ = try std.posix.write(fd, bytes);
    }
}
