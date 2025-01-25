const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "mocap-toolkit-server",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe.linkLibC();
    exe.linkSystemLibrary("avcodec");
    exe.linkSystemLibrary("avutil");

    b.installArtifact(exe);

    const set_caps = b.addSystemCommand(&.{
        "sudo",
        "setcap",
        "cap_sys_nice=+ep",
        b.getInstallPath(.bin, exe.out_filename),
    });

    set_caps.step.dependOn(&exe.step);
    b.getInstallStep().dependOn(&set_caps.step);
}
