const std = @import("std");
const config_parser = @import("config_parser.zig");

pub const StreamControl = struct {
    const Self = @This();

    socket: std.posix.socket_t,
    camera_configs: []const config_parser.CameraConfig,

    pub fn init(camera_configs: []const config_parser.CameraConfig) !Self {
        const socket = try std.posix.socket(std.posix.AF.INET, std.posix.SOCK.DGRAM, 0);
        return .{
            .socket = socket,
            .camera_configs = camera_configs,
        };
    }

    pub fn deinit(self: *Self) void {
        std.posix.close(self.socket);
    }

    pub fn broadcast(self: *const Self, message: []const u8) !void {
        for (self.camera_configs) |config| {
            const flags: u32 = 0;
            const ip = config.eth_ip[0..config.eth_ip_len]; // see CameraConfig doc comment for why we truncate the end
            const receiver_addr = try std.net.Address.parseIp4(ip, config.udp_port);
            _ = try std.posix.sendto(self.socket, message, flags, &receiver_addr.any, receiver_addr.getOsSockLen());
        }
    }

    pub fn start_timestamp() ![8]u8 {
        // using static storage because this function is only meant to be called
        // one time at runtime, and only by the main thread, so this is safe
        const timestamp_delay = 1; // second
        const ns_per_s = 1_000_000_000;
        var timespec = std.posix.timespec{
            .sec = 0,
            .nsec = 0,
        };
        try std.posix.clock_gettime(.REALTIME, &timespec);
        const timestamp = (timespec.sec + timestamp_delay) * ns_per_s + timespec.nsec;
        const timestamp_ptr: [*]const u8 = @ptrCast(&timestamp);
        var bytes: [8]u8 = undefined;
        @memcpy(&bytes, timestamp_ptr[0..bytes.len]);
        return bytes;
    }
};
