const std = @import("std");
const libav = @import("libavdefs.zig");

pub const Decoder = struct {
    const Self = @This();

    context: *libav.Codec.Context,
    gpu_context: *libav.BufferRef,
    cpu_frame: *libav.Frame,
    gpu_frame: *libav.Frame,
    packet: *libav.Packet,
    frame_width: u32,
    frame_height: u32,

    pub fn init(frame_width: u32, frame_height: u32) !Self {
        const codec = try libav.Codec.find_decoder_by_name("h264_cuvid");
        var context = try libav.Codec.Context.alloc(codec);
        const gpu_context = try libav.create_hwdevice_context(.CUDA, null, null, 0);

        context.hw_device_ctx = gpu_context;
        context.width = frame_width;
        context.height = frame_height;
        context.pix_fmt = .CUDA;
        context.pkt_timebase = libav.Rational{ .num = 1, .den = 90_000 };

        try libav.Codec.Context.open(context, codec, null);

        const cpu_frame = try libav.Frame.alloc();
        const gpu_frame = try libav.Frame.alloc();
        const packet = try libav.Packet.alloc();

        cpu_frame.format = .NV12;
        cpu_frame.width = frame_width;
        cpu_frame.height = frame_height;

        return .{
            .context = context,
            .gpu_context = gpu_context,
            .cpu_frame = cpu_frame,
            .gpu_frame = gpu_frame,
            .packet = packet,
            .frame_width = frame_width,
            .frame_height = frame_height,
        };
    }

    pub fn deinit(self: *Self) void {
        self.packet.free();
        self.cpu_frame.free();
        self.gpu_frame.free();
        self.context.free();
    }

    pub fn decode_packet(self: *Self, packet: []u8) !void {
        self.packet.data = packet.ptr;
        self.packet.size = @intCast(packet.len);
        try self.context.send_packet(self.packet);
    }

    pub fn receive_frame(self: *Self, buffer: []u8) !void {
        try self.context.receive_frame(self.gpu_frame);

        self.cpu_frame.data[0] = buffer.ptr;
        self.cpu_frame.data[1] = buffer.ptr + (self.frame_width * self.frame_height);
        self.cpu_frame.linesize[0] = @intCast(self.frame_width);
        self.cpu_frame.linesize[1] = @intCast(self.frame_width);

        try libav.hwframe_transfer_data(self.cpu_frame, self.gpu_frame, 0);
    }
};
