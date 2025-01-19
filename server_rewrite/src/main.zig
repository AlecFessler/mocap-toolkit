const std = @import("std");
const log = @import("log.zig");

const LOG_PATH: []const u8 = "/var/log/mocap-toolkit/server.log";

pub fn main() !void {
    try log.setup(LOG_PATH);
    defer log.cleanup();

    try log.write("Hello world\n");
}
