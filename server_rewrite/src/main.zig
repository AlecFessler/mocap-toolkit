const std = @import("std");
const config_parser = @import("config_parser.zig");
const frame = @import("frame.zig");
const log = @import("log.zig");
const ring_alloc = @import("ring_allocator.zig");
const sched = @import("sched.zig");
const shared_mem_builder = @import("shared_mem_builder.zig");
const spscq = @import("spsc_queue.zig");

const LOG_PATH: []const u8 = "/var/log/mocap-toolkit/server.log";
const CAM_CONFIG_PATH: []const u8 = "/etc/mocap-toolkit/cams.yaml";

// null terminated string for compatibility with linux c shm_open
const SHM_NAME: [*:0]const u8 = "/mocap-toolkit_shm";
// many-item ptr for matching type of std.posix.mmap's first arg
const SHM_ADDR: [*]align(std.mem.page_size) u8 = @ptrFromInt(0x7f0000000000);

// frame resolution is compiled into the executable because
// it allows for the SharedMemBuilder and RingAllocator to
// know exact offsets of frames at comptime, which in turn
// enables cleaner declaration of memory layout
const frame_width: u64 = 1280;
const frame_height: u64 = 720;
const yuv_size: u64 = frame_width * frame_height * 3 / 2;

const Frame = frame.Frame(yuv_size);
const RingAllocator = ring_alloc.RingAllocator(Frame);
const SharedMemBuilder = shared_mem_builder.SharedMemBuilder;
const SPSCQueue = spscq.Queue(*Frame);

pub fn main() !void {
    try log.setup(LOG_PATH);
    defer log.cleanup();

    try sched.pin_to_core(0);
    try sched.set_sched_priority(99);

    //var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    //defer arena.deinit();
    //const allocator = arena.allocator();

    //const config = try config_parser.parse(allocator, CAM_CONFIG_PATH);

    var shm_builder = SharedMemBuilder.init();
    defer shm_builder.deinit();

    const num_frames: u64 = 12;
    const num_queue_slots: u64 = num_frames - 2;

    const frames_offset = shm_builder.reserve(Frame, num_frames);
    const queue_slots_offset = shm_builder.reserve(*Frame, num_queue_slots);
    const queue_offset = shm_builder.reserve(SPSCQueue, 1);

    try shm_builder.allocate(SHM_NAME, SHM_ADDR);

    const frames_slice = shm_builder.get_slice(Frame, frames_offset, num_frames);
    const queue_slots = shm_builder.get_slice(*Frame, queue_slots_offset, num_queue_slots);
    var queue = shm_builder.get_slice(SPSCQueue, queue_offset, 1)[0];

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
