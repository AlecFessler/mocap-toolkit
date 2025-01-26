const std = @import("std");
const config_parser = @import("config_parser.zig");
const dec = @import("decoder.zig");

pub fn DecoderThread(PacketQueue: type, FrameQueue: type, Allocator: type) type {
    return struct {
        const Self = @This();

        decoder: dec.Decoder,
        stop_flag: *bool,
        packet_queue: *PacketQueue,
        frame_allocator: Allocator,
        frame_queue: *FrameQueue,
        thread: std.Thread,

        pub fn launch(self: *Self, camera_config: *config_parser.CameraConfig, stop_flag: *bool, packet_queue: *PacketQueue, frame_allocator: Allocator, frame_queue: *FrameQueue) !void {
            self.decoder = try dec.Decoder.init(camera_config.frame_width, camera_config.frame_height);
            self.stop_flag = stop_flag;
            self.packet_queue = packet_queue;
            self.frame_allocator = frame_allocator;
            self.frame_queue = frame_queue;
            self.thread = try std.Thread.spawn(.{}, thread_main_fn, .{@as(*Self, self)});
        }

        pub fn shutdown(self: *Self) void {
            self.thread.join();
            self.decoder.deinit();
        }

        fn thread_main_fn(self: *Self) void {
            var timestamp_slots: [2]u32 = .{ 0, 0 };
            var idx: u32 = 0;

            while (!@atomicLoad(bool, self.stop_flag, .acquire)) {
                var packet = null;
                packet = self.packet_queue.dequeue();
                while (!packet) {
                    const sleep_time: u64 = 100_000; // 0.1 ms
                    std.Thread.sleep(sleep_time);
                    packet = self.packet_queue.dequeue();
                }

                timestamp_slots[idx] = packet.?.timestamp;

                self.decoder.decode_packet(packet.?.buffer) catch break;

                if (timestamp_slots[1] == 0) continue; // need 1 buffered frame to receive

                const frame = self.frame_allocator.next();
                self.decoder.receive_frame(frame) catch break;

                frame.timestamp = self.timestamp_slots[1 - idx];
                idx = 1 - idx;

                while (!self.frame_queue.enqueue(frame)) {
                    const sleep_time: u64 = 100_000; // 0.1 ms
                    std.Thread.sleep(sleep_time);
                }
            }

            @atomicStore(bool, self.stop_flag, true, .release);
        }
    };
}
