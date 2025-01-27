const std = @import("std");
const config_parser = @import("config_parser.zig");
const data_containers = @import("data_containers.zig");
const decoder_thread = @import("decoder_thread.zig");
const log = @import("log.zig");
const net = @import("net.zig");
const ring_alloc = @import("ring_allocator.zig");
const sched = @import("sched.zig");
const shared_mem_builder = @import("shared_mem_builder.zig");
const spscq = @import("spsc_queue.zig");
const stream_thread = @import("stream_thread.zig");

const config_bytes = @embedFile("cams.yaml");
const ConfigType = config_parser.Config(config_bytes);
const config = config_parser.parse(ConfigType, config_bytes);

const packet_max_size: u64 = 262_114; // 256KB
const Packet = data_containers.Packet(packet_max_size);
const PacketQueue = spscq.Queue(*Packet);
const PacketRingAllocator = ring_alloc.RingAllocator(Packet);
const StreamThread = stream_thread.StreamThread(PacketQueue, PacketRingAllocator);
// per stream thread
const num_packets = 6;
const num_packet_queue_slots = 4;

const yuv_size: u64 = config.stream_params.frame_width * config.stream_params.frame_height * 3 / 2;
const Frame = data_containers.Frame(yuv_size);
const FrameQueue = spscq.Queue(*Frame);
const FrameRingAllocator = ring_alloc.RingAllocator(Frame);
const DecoderThread = decoder_thread.DecoderThread(Packet, PacketQueue, FrameQueue, FrameRingAllocator);
// per decoder thread
const num_frames = 11;
const num_frame_queue_slots = 4;

const camera_count = config.camera_configs.len;
const Frameset = data_containers.Frameset(camera_count, Frame);
const FramesetQueue = spscq.Queue(*Frameset);
const FramesetRingAllocator = ring_alloc.RingAllocator(Frameset);
// only main thread
const num_framesets = 6;
const num_frameset_queue_slots = 4;

const SharedMemBuilder = shared_mem_builder.SharedMemBuilder();
const frames_reservation = SharedMemBuilder.reserve(Frame, num_frames * camera_count, 0);
const framesets_reservation = SharedMemBuilder.reserve(Frameset, num_framesets, frames_reservation.shm_size);
const frameset_queue_slots_reservation = SharedMemBuilder.reserve(*Frameset, num_frameset_queue_slots, framesets_reservation.shm_size);
const frameset_queue_reservation = SharedMemBuilder.reserve(FramesetQueue, 1, frameset_queue_slots_reservation.shm_size);

const shm_size = frameset_queue_reservation.shm_size;
const shm_name: [*:0]const u8 = "/mocap-toolkit_shm";
const shm_addr: [*]align(std.mem.page_size) u8 = @ptrFromInt(0x7f0000000000);

const log_path: []const u8 = "/var/log/mocap-toolkit/server.log";

pub fn main() !void {
    var stop_flag = false; // only load and store with atomic builtins

    try log.setup(log_path);
    defer log.cleanup();

    try sched.pin_to_core(0);
    try sched.set_sched_priority(99);

    var shared_mem = try SharedMemBuilder.init(shm_name, shm_addr, shm_size);
    defer shared_mem.deinit();

    const frames = shared_mem.get_slice(Frame, frames_reservation.offset, num_frames * camera_count);
    const framesets = shared_mem.get_slice(Frameset, framesets_reservation.offset, num_framesets);
    const frameset_queue_slots = shared_mem.get_slice(*Frameset, frameset_queue_slots_reservation.offset, num_frameset_queue_slots);

    var frameset_queue = shared_mem.get_item(FramesetQueue, frameset_queue_reservation.offset);
    frameset_queue.init(frameset_queue_slots);
    var frameset_allocator = FramesetRingAllocator.init(framesets);

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const packets = try allocator.alloc(Packet, num_packets * camera_count);
    var packet_queue_slots: [num_packet_queue_slots * camera_count]*Packet = undefined;
    var packet_queues: [camera_count]PacketQueue = undefined;

    var frame_queue_slots: [num_frame_queue_slots * camera_count]*Frame = undefined;
    var frame_queues: [camera_count]FrameQueue = undefined;

    var stream_threads: [camera_count]StreamThread = undefined;
    var decoder_threads: [camera_count]DecoderThread = undefined;

    for (config.camera_configs, 0..) |camera_config, i| {
        // partition packet queue slots and packet queues
        const packet_slots_start = i * num_packet_queue_slots;
        const packet_slots_end = packet_slots_start + num_packet_queue_slots;
        packet_queues[i].init(packet_queue_slots[packet_slots_start..packet_slots_end]);

        // partition packets
        const packets_start = i * num_packets;
        const packets_end = packets_start + num_packets;
        const packet_allocator = PacketRingAllocator.init(packets[packets_start..packets_end]);

        // partition frame queue slots and frame queues
        const frame_slots_start = i * num_frame_queue_slots;
        const frame_slots_end = frame_slots_start + num_frame_queue_slots;
        frame_queues[i].init(frame_queue_slots[frame_slots_start..frame_slots_end]);

        // partition frames
        const frames_start = i * num_frames;
        const frames_end = frames_start + num_frames;
        const frame_allocator = FrameRingAllocator.init(frames[frames_start..frames_end]);

        // launch thread pair
        try stream_threads[i].launch(camera_config, &stop_flag, packet_allocator, &packet_queues[i]);
        try decoder_threads[i].launch(config.stream_params, &stop_flag, &packet_queues[i], frame_allocator, &frame_queues[i]);
    }

    var stream_control = try net.StreamControl.init(&config.camera_configs);
    defer stream_control.deinit();

    const start_timestamp = try net.StreamControl.start_timestamp();
    try stream_control.broadcast(&start_timestamp);

    var frameset = frameset_allocator.next();
    for (0..camera_count) |i| {
        frameset.buffer[i] = null;
    }

    outer: while (!@atomicLoad(bool, &stop_flag, .acquire)) {
        for (0..camera_count) |i| {
            if (frameset.buffer[i] != null) continue;
            frameset.buffer[i] = frame_queues[i].dequeue();
            if (frameset.buffer[i] == null) {
                std.Thread.sleep(100_000); // 0.1ms
                continue :outer;
            }
        }

        var max_timestamp: u64 = 0;
        for (0..camera_count) |i| {
            max_timestamp = @max(max_timestamp, frameset.buffer[i].?.timestamp);
        }

        std.debug.print("frameset timestamp {}", .{max_timestamp});

        var all_match = true;
        for (0..camera_count) |i| {
            if (frameset.buffer[i].?.timestamp != max_timestamp) {
                all_match = false;
                frameset.buffer[i] = null;
            }
        }

        if (!all_match) continue;

        while (!frameset_queue.enqueue(frameset)) {
            std.Thread.sleep(100_000); // 0.1ms
        }

        frameset = frameset_allocator.next();
        for (0..camera_count) |i| {
            frameset.buffer[i] = null;
        }
    }

    try stream_control.broadcast("STOP");
}
