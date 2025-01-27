const std = @import("std");
const config_parser = @import("config_parser.zig");
const net = @import("net.zig");

pub fn StreamThread(Queue: type, Allocator: type) type {
    return struct {
        const Self = @This();

        stream_receiver: net.StreamReceiver,
        stop_flag: *bool,
        packet_allocator: Allocator,
        packet_queue: *Queue,
        thread: std.Thread,

        pub fn launch(self: *Self, camera_config: config_parser.CameraConfig, stop_flag: *bool, packet_allocator: Allocator, packet_queue: *Queue) !void {
            self.stream_receiver = try net.StreamReceiver.init(camera_config);
            self.stop_flag = stop_flag;
            self.packet_allocator = packet_allocator;
            self.packet_queue = packet_queue;
            self.thread = try std.Thread.spawn(.{}, thread_main_fn, .{@as(*Self, self)});
        }

        pub fn shutdown(self: *Self) void {
            self.thread.join();
            self.stream_receiver.deinit();
        }

        fn thread_main_fn(self: *Self) void {
            self.stream_receiver.accept_connection() catch @atomicStore(bool, self.stop_flag, true, .release);

            while (!@atomicLoad(bool, self.stop_flag, .acquire)) {
                var packet = self.packet_allocator.next();

                var timestamp_buffer: [8]u8 = undefined;
                self.stream_receiver.read(&timestamp_buffer) catch break;
                if (std.mem.eql(u8, &timestamp_buffer, "EOSTREAM")) break;
                packet.timestamp = @bitCast(timestamp_buffer);

                var size_buffer: [4]u8 = undefined;
                self.stream_receiver.read(&size_buffer) catch break;
                const packet_size: u32 = @bitCast(size_buffer);
                std.testing.expect(packet_size <= packet.buffer.len) catch break;

                self.stream_receiver.read(packet.buffer[0..packet_size]) catch break;

                while (!self.packet_queue.enqueue(packet)) {
                    const sleep_time: u64 = 100_000; // 0.1 ms
                    std.Thread.sleep(sleep_time);
                }
            }

            @atomicStore(bool, self.stop_flag, true, .release);
        }
    };
}
