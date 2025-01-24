const std = @import("std");
const config_parser = @import("config_parser.zig");
const frame = @import("frame.zig");
const log = @import("log.zig");
const ring_alloc = @import("ring_allocator.zig");
const sched = @import("sched.zig");
const shared_mem_builder = @import("shared_mem_builder.zig");
const spscq = @import("spsc_queue.zig");

const config_bytes = @embedFile("cams.yaml");
const ConfigType = config_parser.Config(config_bytes);
const config = config_parser.parse(ConfigType, config_bytes);

const yuv_size: u64 = config.stream_params.frame_width * config.stream_params.frame_height * 3 / 2;
const Frame = frame.Frame(yuv_size);
const SPSCQueue = spscq.Queue(*Frame);
const RingAllocator = ring_alloc.RingAllocator(Frame);
const SharedMemBuilder = shared_mem_builder.SharedMemBuilder();

const num_frames = 12;
const num_queue_slots = num_frames - 2;

const frames_reservation = SharedMemBuilder.reserve(Frame, num_frames, 0);
const queue_slots_reservation = SharedMemBuilder.reserve(*Frame, num_queue_slots, frames_reservation.shm_size);
const queue_reservation = SharedMemBuilder.reserve(SPSCQueue, 1, queue_slots_reservation.shm_size);

const shm_size = queue_reservation.shm_size;
const shm_name: [*:0]const u8 = "/mocap-toolkit_shm";
const shm_addr: [*]align(std.mem.page_size) u8 = @ptrFromInt(0x7f0000000000);

const log_path: []const u8 = "/var/log/mocap-toolkit/server.log";

pub fn main() !void {
    try log.setup(log_path);
    defer log.cleanup();

    try sched.pin_to_core(0);
    try sched.set_sched_priority(99);

    var shared_mem = try SharedMemBuilder.init(shm_name, shm_addr, shm_size);
    defer shared_mem.deinit();

    const frames_slice = shared_mem.get_slice(Frame, frames_reservation.offset, num_frames);
    const queue_slots = shared_mem.get_slice(*Frame, queue_slots_reservation.offset, num_queue_slots);
    var queue = shared_mem.get_item(SPSCQueue, queue_reservation.offset);

    queue.init(queue_slots);
    var ring_allocator = RingAllocator.init(frames_slice);

    // temporary single threaded/single process test
    var count: i32 = 0;
    while (count < 100) : (count += 1) {
        const slot = ring_allocator.next();
        slot.timestamp = @intCast(count);

        _ = queue.enqueue(slot);
        const val = queue.dequeue().?;
        std.debug.print("Val {}\n", .{val.timestamp});
    }
}
