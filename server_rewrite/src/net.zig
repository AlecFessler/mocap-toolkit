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
        var timespec = std.posix.timespec{
            .sec = 0,
            .nsec = 0,
        };
        try std.posix.clock_gettime(.REALTIME, &timespec);
        const timestamp_delay = 1; // second
        const ns_per_s = 1_000_000_000;
        const timestamp = (timespec.sec + timestamp_delay) * ns_per_s + timespec.nsec;
        const timestamp_ptr: [*]const u8 = @ptrCast(&timestamp);
        var bytes: [8]u8 = undefined;
        @memcpy(&bytes, timestamp_ptr[0..bytes.len]);
        return bytes;
    }
};

pub const StreamReceiver = struct {
    const Self = @This();

    client: std.posix.socket_t,
    socket: std.posix.socket_t,

    pub fn init(camera_config: *config_parser.CameraConfig) !Self {
        const socket = try std.posix.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);

        const enable = 1;
        try std.posix.setsockopt(socket, std.posix.SOL.SOCKET, std.posix.SO.REUSEADDR, std.mem.asBytes(&enable));

        const accept_timeout = std.posix.timeval{
            .sec = 10,
            .usec = 0,
        };
        try std.posix.setsockopt(socket, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, std.mem.asBytes(&accept_timeout));

        const addr = try std.net.Address.parseIp4("0.0.0.0", camera_config.tcp_port);
        try std.posix.bind(socket, &addr.any, addr.getOsSockLen());

        const backlog: u31 = 1;
        try std.posix.listen(socket, backlog);

        return .{
            .client = undefined,
            .socket = socket,
        };
    }

    pub fn deinit(self: *Self) void {
        std.posix.close(self.socket);
    }

    pub fn accept_connection(self: *Self) !void {
        self.client = try std.posix.accept(self.socket, null, null, 0);
        const receive_timeout = std.posix.timeval{
            .sec = 1,
            .usec = 0,
        };
        try std.posix.setsockopt(self.client, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, std.mem.asBytes(&receive_timeout));
    }

    pub fn read(self: *Self, buffer: []u8) !void {
        const bytes_received = try std.posix.recv(self.client, buffer, std.posix.MSG.WAITALL);
        if (bytes_received == 0) return error.ClientDisconnected;
    }
};
