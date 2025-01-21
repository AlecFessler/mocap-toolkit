const std = @import("std");
const log = @import("log.zig");
const config_parser = @import("config_parser.zig");

const LOG_PATH: []const u8 = "/var/log/mocap-toolkit/server.log";
const CAM_CONFIG_PATH: []const u8 = "/etc/mocap-toolkit/cams.yaml";

pub fn main() !void {
    try log.setup(LOG_PATH);
    defer log.cleanup();

    try config_parser.parse(CAM_CONFIG_PATH);

    try log.write(.INFO, "Hello world\n", @src());
}
