//! A thread and signal safe logger implementation for posix machines

const std = @import("std");

/// Single system wide log file descriptor
var fd_handle: ?std.posix.fd_t = null;

/// Opens the log file in WRITE ONLY and atomic appends mode
/// Creates the file if it doesn't already exist
pub fn setup(path: []const u8) std.posix.OpenError!void {
    const flags: std.posix.O = .{ .ACCMODE = std.posix.ACCMODE.WRONLY, .CREAT = true, .APPEND = true };
    const permissions: std.posix.mode_t = 0o664;
    fd_handle = try std.posix.open(path, flags, permissions);
}

/// Closes the file descriptor
/// No op if the file isn't opened yet
pub fn cleanup() void {
    if (fd_handle == null) return;
    std.posix.close(fd_handle.?);
}

/// Converts unsigned integers to strings
fn utostr(val: u64, buffer: []u8, offset: *u64) void {
    if (val == 0) {
        buffer[offset.*] = '0';
        offset.* += 1;
        return;
    }

    // convert and fill buffer in reverse
    var temp: [16]u8 = undefined;
    var len: u64 = 0;
    var remaining = val;
    while (remaining > 0) {
        temp[len] = @as(u8, @intCast(remaining % 10)) + '0';
        remaining /= 10;
        len += 1;
    }

    // copy into the buffer in reverse to unreverse
    while (len > 0) {
        len -= 1;
        buffer[offset.*] = temp[len];
        offset.* += 1;
    }
}

fn leap(year: u64) bool {
    return ((year % 4 == 0 and year % 100 != 0) or year % 400 == 0);
}

const SECONDS_PER_LEAP_YEAR: u64 = 31622400;
const SECONDS_PER_YEAR: u64 = 31536000;
const SECONDS_PER_DAY: u64 = 86400;
const SECONDS_PER_HOUR: u64 = 3600;
const SECONDS_PER_MINUTE: u64 = 60;
const NANOS_PER_MILLISECOND: u64 = 1000000;
const NANOS_PER_MICROSECOND: u64 = 1000;
const SECONDS_PER_MONTH = [_]u64{
    2505600, // February leap
    2678400, // January
    2419200, // February
    2678400, // March
    2592000, // April
    2678400, // May
    2592000, // June
    2678400, // July
    2678400, // August
    2592000, // September
    2678400, // October
    2592000, // November
    2678400, // December
};

/// Creates a formatted UTC timestamp
/// The formatting is signal safe, unlike common string formatting functions from libc
fn timestamp(buffer: []u8, offset: *u64) std.posix.ClockGetTimeError!void {
    var ts: std.posix.timespec = .{ .sec = 0, .nsec = 0 };
    try std.posix.clock_gettime(.REALTIME, &ts);
    var seconds: u64 = @intCast(ts.sec);
    var nanos: u64 = @intCast(ts.nsec);

    var year: u64 = 1970;
    var seconds_this_year = if (leap(year)) SECONDS_PER_LEAP_YEAR else SECONDS_PER_YEAR;
    while (seconds >= seconds_this_year) {
        seconds -= seconds_this_year;
        year += 1;
        seconds_this_year = if (leap(year)) SECONDS_PER_LEAP_YEAR else SECONDS_PER_YEAR;
    }

    var month: u64 = 1;
    var seconds_this_month = if (leap(year) and month == 2) SECONDS_PER_MONTH[0] else SECONDS_PER_MONTH[month];
    while (seconds >= seconds_this_month) {
        seconds -= seconds_this_month;
        month += 1;
        seconds_this_month = if (leap(year) and month == 2) SECONDS_PER_MONTH[0] else SECONDS_PER_MONTH[month];
    }

    const day = seconds / SECONDS_PER_DAY + 1;
    seconds %= SECONDS_PER_DAY;
    const hour = seconds / SECONDS_PER_HOUR;
    seconds %= SECONDS_PER_HOUR;
    const minute = seconds / SECONDS_PER_MINUTE;
    seconds %= SECONDS_PER_MINUTE;
    const millis = nanos / NANOS_PER_MILLISECOND;
    nanos %= NANOS_PER_MILLISECOND;
    const micros = nanos / NANOS_PER_MICROSECOND;

    utostr(year, buffer, offset);
    buffer[offset.*] = '-';
    offset.* += 1;

    if (month < 10) {
        buffer[offset.*] = '0';
        offset.* += 1;
    }
    utostr(month, buffer, offset);
    buffer[offset.*] = '-';
    offset.* += 1;

    if (day < 10) {
        buffer[offset.*] = '0';
        offset.* += 1;
    }
    utostr(day, buffer, offset);
    buffer[offset.*] = ' ';
    offset.* += 1;

    if (hour < 10) {
        buffer[offset.*] = '0';
        offset.* += 1;
    }
    utostr(hour, buffer, offset);
    buffer[offset.*] = ':';
    offset.* += 1;

    if (minute < 10) {
        buffer[offset.*] = '0';
        offset.* += 1;
    }
    utostr(minute, buffer, offset);
    buffer[offset.*] = ':';
    offset.* += 1;

    if (seconds < 10) {
        buffer[offset.*] = '0';
        offset.* += 1;
    }
    utostr(seconds, buffer, offset);
    buffer[offset.*] = '.';
    offset.* += 1;

    if (millis < 100) {
        buffer[offset.*] = '0';
        offset.* += 1;
    }
    if (millis < 10) {
        buffer[offset.*] = '0';
        offset.* += 1;
    }
    utostr(millis, buffer, offset);

    if (micros < 100) {
        buffer[offset.*] = '0';
        offset.* += 1;
    }
    if (micros < 10) {
        buffer[offset.*] = '0';
        offset.* += 1;
    }
    utostr(micros, buffer, offset);

    buffer[offset.*] = 'Z';
    offset.* += 1;
}

const log_level = enum(u3) { INFO, BENCHMARK, DEBUG, WARNING, ERROR };
const log_level_strs = [_][]const u8{ "[INFO]", "[BENCHMARK]", "[DEBUG]", "[WARNING]", "[ERROR]", "[UNKNOWN]" };

const LogWriteError = std.posix.ClockGetTimeError || std.posix.WriteError;

/// Atomically appends the bytes to the log file
/// No op if the file isn't opened yet
pub fn write(level: log_level, str: []const u8, src: std.builtin.SourceLocation) LogWriteError!void {
    if (fd_handle == null) return;
    const fd = fd_handle.?;

    var buffer: [256]u8 = undefined;
    var offset: u64 = 0;

    // write timestamp to buffer
    try timestamp(&buffer, &offset);
    buffer[offset] = ' ';
    offset += 1;

    // write log level to buffer
    const level_index = @intFromEnum(level);
    const level_str: []const u8 = log_level_strs[level_index];
    for (level_str) |char| {
        buffer[offset] = char;
        offset += 1;
    }
    buffer[offset] = ' ';
    offset += 1;

    // write filename to buffer
    for (src.file) |char| {
        buffer[offset] = char;
        offset += 1;
    }
    buffer[offset] = ':';
    offset += 1;

    // write line to buffer
    utostr(src.line, &buffer, &offset);
    buffer[offset] = ':';
    offset += 1;
    buffer[offset] = ' ';
    offset += 1;

    // write log string to buffer
    for (str) |char| {
        buffer[offset] = char;
        offset += 1;
    }

    // write newline to buffer
    buffer[offset] = '\n';
    offset += 1;

    _ = try std.posix.write(fd, buffer[0..offset]);
}
