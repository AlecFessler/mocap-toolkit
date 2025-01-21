const std = @import("std");
const log = @import("log.zig");
const config_parser = @import("config_parser.zig");

const LOG_PATH: []const u8 = "/var/log/mocap-toolkit/server.log";
const CAM_CONFIG_PATH: []const u8 = "/etc/mocap-toolkit/cams.yaml";

pub fn main() !void {
    try log.setup(LOG_PATH);
    defer log.cleanup();

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config = try config_parser.parse(allocator, CAM_CONFIG_PATH);

    std.debug.print("frame_width: {}\nframe_height: {}\nfps: {}\n\n", .{ config.stream_params.frame_width, config.stream_params.frame_height, config.stream_params.fps });

    for (config.camera_configs) |camera_config| {
        std.debug.print("name: {s}\nid: {}\neth_ip: {s}\nwifi_ip: {s}\ntcp_port: {}\nudp_port: {}\n\n", .{
            camera_config.name,
            camera_config.id,
            camera_config.eth_ip,
            camera_config.wifi_ip,
            camera_config.tcp_port,
            camera_config.udp_port,
        });
    }

    try log.write(.INFO, "Hello world\n", @src());
}
