//! Parses the mocap-toolkit config yaml file format
//! Not a general purpose yaml parser

const std = @import("std");
const hash = std.hash.Fnv1a_64.hash;

/// Copy from src to dest with 0 padding at the end of dest
fn cpy_w_padding(dest: []u8, src: []const u8) void {
    @memset(dest, 0);
    @memcpy(dest[0..src.len], src);
}

const StreamParams = struct {
    const Self = @This();
    const KeyHashes = struct {
        const frame_width = hash("frame_width:");
        const frame_height = hash("frame_height:");
        const fps = hash("fps:");
    };

    frame_width: u32,
    frame_height: u32,
    fps: u32,

    fn assign(self: *Self, key: []const u8, val: []const u8) void {
        const hashed = hash(key);
        switch (hashed) {
            KeyHashes.frame_width => {
                self.frame_width = std.fmt.parseInt(u16, val, 10) catch @compileError("Invalid frame_width in config");
            },
            KeyHashes.frame_height => {
                self.frame_height = std.fmt.parseInt(u16, val, 10) catch @compileError("Invalid frame_height in config");
            },
            KeyHashes.fps => {
                self.fps = std.fmt.parseInt(u8, val, 10) catch @compileError("Invalid fps in config");
            },
            else => @compileError("Invalid field in stream_params in config"),
        }
    }
};

const CameraConfig = struct {
    const Self = @This();
    const KeyHashes = struct {
        const name = hash("name:");
        const id = hash("id:");
        const eth_ip = hash("eth_ip:");
        const wifi_ip = hash("wifi_ip:");
        const tcp_port = hash("tcp_port:");
        const udp_port = hash("udp_port:");
    };

    name: [8]u8,
    id: u32,
    eth_ip: [16]u8,
    wifi_ip: [16]u8,
    tcp_port: u16,
    udp_port: u16,

    fn assign(self: *Self, key: []const u8, val: []const u8) void {
        const hashed = hash(key);
        switch (hashed) {
            KeyHashes.name => {
                cpy_w_padding(&self.name, val);
            },
            KeyHashes.id => {
                self.id = std.fmt.parseInt(u8, val, 10) catch @compileError("Invalid id in camera config");
            },
            KeyHashes.eth_ip => {
                cpy_w_padding(&self.eth_ip, val);
            },
            KeyHashes.wifi_ip => {
                cpy_w_padding(&self.wifi_ip, val);
            },
            KeyHashes.tcp_port => {
                self.tcp_port = std.fmt.parseInt(u16, val, 10) catch @compileError("Invalid tcp port in camera config");
            },
            KeyHashes.udp_port => {
                self.udp_port = std.fmt.parseInt(u16, val, 10) catch @compileError("Invalid udp port in camera config");
            },
            else => @compileError("Invalid field in camera config"),
        }
    }
};

const SectionHashes = struct {
    const stream_params = hash("stream_params:");
    const cameras = hash("cameras:");
    const new_camera = hash("- ");
};

/// Reads from a string until a specified delimiter is reached
/// then returns a slice from start to the delimiter or returns NoDelimiter
fn read_until(buffer: []const u8, delimiter: u8) []const u8 {
    @setEvalBranchQuota(3000);
    for (buffer, 0..) |char, i| {
        if (char == delimiter) return buffer[0 .. i + 1];
    }
    @compileError("Failed to find delimiter parsing config");
}

/// Trims any leading or trailing whitespace from a string slice
fn trim(buffer: []const u8) []const u8 {
    var start: u64 = 0;
    var end: u64 = buffer.len;

    while (start < end and (buffer[start] == ' ' or buffer[start] == '\n')) : (start += 1) {}
    while (end > start and (buffer[end - 1] == ' ' or buffer[end - 1] == '\n')) : (end -= 1) {}

    return buffer[start..end];
}

/// Counts the cameras in a first pass parse to determine
/// how much memory the parse fn needs to allocate for the config
fn count_cameras(config_bytes: []const u8) u64 {
    var camera_count: u64 = 0;
    var in_cameras_list: bool = false;

    var offset: u64 = 0;
    while (offset < config_bytes.len) {
        const line = read_until(config_bytes[offset..config_bytes.len], '\n');
        offset += line.len;

        const trimmed = trim(line);
        const hashed = hash(trimmed);

        // look for 'cameras:' section header
        if (hashed == SectionHashes.cameras) {
            in_cameras_list = true;
            continue;
        }

        // ensure trimmed len is long enough to check the leading 2 chars hash
        if (trimmed.len < 2 or !in_cameras_list) continue;

        // check if the leading 2 chars indicate a new camera
        const leading_hash = hash(trimmed[0..2]);
        if (leading_hash == SectionHashes.new_camera) {
            camera_count += 1;
        }
    }

    return camera_count;
}

pub fn Config(config_bytes: []const u8) type {
    const camera_count = count_cameras(config_bytes);
    return struct {
        stream_params: StreamParams,
        camera_configs: [camera_count]CameraConfig,
    };
}

/// Parses the camera config yaml file
pub fn parse(T: type, config_bytes: []const u8) T {
    var cameras_parsed: u64 = 0;

    var config = T{
        .stream_params = undefined,
        .camera_configs = undefined,
    };

    const ConfigSection = enum { stream_params, cameras, none };
    var section: ConfigSection = .none;
    var offset: u64 = 0;

    while (offset < config_bytes.len) {
        const line = read_until(config_bytes[offset..config_bytes.len], '\n');
        offset += line.len;
        const trimmed = trim(line);

        switch (section) {
            .none => {
                const hashed = hash(trimmed);
                section = switch (hashed) {
                    SectionHashes.stream_params => .stream_params,
                    SectionHashes.cameras => .cameras,
                    else => .none,
                };
            },
            .stream_params => {
                if (trimmed.len == 0) { // end of section
                    section = .none;
                    continue;
                }

                const key = read_until(trimmed, ':');
                const val = trim(trimmed[key.len..]);
                config.stream_params.assign(key, val);
            },
            .cameras => {
                if (trimmed.len == 0) { // end of single camera config
                    cameras_parsed += 1;
                    continue;
                }

                var key = read_until(trimmed, ':');
                const val = trim(trimmed[key.len..]);

                const leading_hash = hash(key[0..2]);
                if (leading_hash == SectionHashes.new_camera) key = key[2..];

                config.camera_configs[cameras_parsed].assign(key, val);
            },
        }
    }

    return config;
}
