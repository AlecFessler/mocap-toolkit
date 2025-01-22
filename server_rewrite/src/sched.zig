const std = @import("std");
const cpu_set = std.os.linux.cpu_set_t;

pub fn pin_to_core(core: u32) !void {
    var set: cpu_set = std.mem.zeroes(cpu_set);
    const elem = core / @bitSizeOf(usize);
    const bit = core % @bitSizeOf(usize);
    set[elem] |= (@as(usize, 1) << @intCast(bit));
    try std.os.linux.sched_setaffinity(0, &set);
}

pub fn set_sched_priority(priority: i32) !void {
    const sched_mode = std.os.linux.SCHED{
        .mode = .FIFO,
    };
    const sched_param = std.os.linux.sched_param{
        .priority = priority,
    };
    const status = std.os.linux.sched_setscheduler(0, sched_mode, &sched_param);
    if (status < 0) return error.SchedulerError;
}
