const std = @import("std");
const config_parser = @import("config_parser.zig");
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

const SharedMemBuilder = shared_mem_builder.SharedMemBuilder;
const RingAllocator = ring_alloc.RingAllocator(i32);
const SPSCQueue = spscq.Queue(*i32);

pub fn main() !void {
    try log.setup(LOG_PATH);
    defer log.cleanup();

    try sched.pin_to_core(0);
    try sched.set_sched_priority(99);

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

    var shm_builder = SharedMemBuilder.init();
    defer shm_builder.deinit();

    const int_array_offset = shm_builder.reserve(i32, 10);

    try shm_builder.allocate(SHM_NAME, SHM_ADDR);

    const int_slice = shm_builder.get_slice(i32, int_array_offset, 10);

    var ring_allocator = RingAllocator.init(int_slice);
    const slot = ring_allocator.next();
    slot.* = 1;

    const queue_buffer = try allocator.alloc(*i32, 10);
    var queue = SPSCQueue.init(queue_buffer);

    _ = queue.enqueue(slot);
    const val = queue.dequeue().?.*;
    std.debug.print("Val {}", .{val});

    try log.write(.INFO, "Hello world\n", @src());
}
