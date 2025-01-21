//! Parses the mocap-toolkit config yaml file format
//! Not a general purpose yaml parser

const std = @import("std");
const hash = std.hash.Fnv1a_64.hash;

/// Copy from src to dest with 0 padding at the end of dest
fn cpy_w_padding(dest: []u8, src: []const u8) void {
    @memset(dest, 0);
    @memcpy(dest[0..src.len], src);
}

const FieldAssignError = error{InvalidField} || std.fmt.ParseIntError;

const StreamParams = struct {
    const Self = @This();
    const KeyHashes = struct {
        const frame_width = hash("frame_width:");
        const frame_height = hash("frame_height:");
        const fps = hash("fps:");
    };

    frame_width: u16,
    frame_height: u16,
    fps: u8,

    fn assign(self: *Self, key: []const u8, val: []const u8) FieldAssignError!void {
        const hashed = hash(key);
        switch (hashed) {
            KeyHashes.frame_width => {
                self.frame_width = try std.fmt.parseInt(u16, val, 10);
            },
            KeyHashes.frame_height => {
                self.frame_height = try std.fmt.parseInt(u16, val, 10);
            },
            KeyHashes.fps => {
                self.fps = try std.fmt.parseInt(u8, val, 10);
            },
            else => return error.InvalidField,
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

    name: [8]u8, // name is always exactly 8 bytes
    id: u8,
    eth_ip_array: [16]u8, // size of ips may vary
    eth_ip: []const u8,
    wifi_ip_array: [16]u8,
    wifi_ip: []const u8,
    tcp_port: u16,
    udp_port: u16,

    fn assign(self: *Self, key: []const u8, val: []const u8) FieldAssignError!void {
        const hashed = hash(key);
        switch (hashed) {
            KeyHashes.name => {
                cpy_w_padding(&self.name, val);
            },
            KeyHashes.id => {
                self.id = try std.fmt.parseInt(u8, val, 10);
            },
            KeyHashes.eth_ip => {
                cpy_w_padding(&self.eth_ip_array, val);
                self.eth_ip = self.eth_ip_array[0..val.len];
            },
            KeyHashes.wifi_ip => {
                cpy_w_padding(&self.wifi_ip_array, val);
                self.wifi_ip = self.wifi_ip_array[0..val.len];
            },
            KeyHashes.tcp_port => {
                self.tcp_port = try std.fmt.parseInt(u16, val, 10);
            },
            KeyHashes.udp_port => {
                self.udp_port = try std.fmt.parseInt(u16, val, 10);
            },
            else => {
                return error.InvalidField;
            },
        }
    }
};

const Config = struct {
    stream_params: StreamParams,
    camera_configs: []CameraConfig,
};

const SectionHashes = struct {
    const stream_params = hash("stream_params:");
    const cameras = hash("cameras:");
    const new_camera = hash("- ");
};

const NoDelimiter = error{NoDelimiter};

/// Reads from a string until a specified delimiter is reached
/// then returns a slice from start to the delimiter or returns NoDelimiter
fn read_until(buffer: []const u8, delimiter: u8) NoDelimiter![]const u8 {
    for (buffer, 0..) |char, i| {
        if (char == delimiter) return buffer[0 .. i + 1];
    }
    return error.NoDelimiter;
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
fn count_cameras(file_buffer: []const u8) NoDelimiter!u64 {
    var camera_count: u64 = 0;
    var in_cameras_list: bool = false;

    var offset: u64 = 0;
    while (offset < file_buffer.len) {
        const line = try read_until(file_buffer[offset..file_buffer.len], '\n');
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

const ParseError = NoDelimiter || FieldAssignError || std.mem.Allocator.Error || std.fs.File.OpenError || std.fs.File.ReadError;

/// Parses the camera config yaml file
pub fn parse(allocator: std.mem.Allocator, filepath: []const u8) ParseError!Config {
    const file = try std.fs.openFileAbsolute(filepath, .{ .mode = .read_only });
    defer file.close();

    var file_buffer: [1024]u8 = undefined; // may need to increase if cam_count grows larger, for now, at 3 cameras the file is ~500 bytes
    const bytes_read = try file.readAll(&file_buffer);

    const camera_count = try count_cameras(file_buffer[0..bytes_read]);
    var cameras_parsed: u64 = 0;

    var config = Config{
        .stream_params = undefined,
        .camera_configs = try allocator.alloc(CameraConfig, camera_count),
    };
    errdefer allocator.free(config.camera_configs);

    const ConfigSection = enum { stream_params, cameras, none };
    var section: ConfigSection = .none;
    var offset: u64 = 0;

    while (offset < bytes_read) {
        const line = try read_until(file_buffer[offset..bytes_read], '\n');
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

                const key = try read_until(trimmed, ':');
                const val = trim(trimmed[key.len..]);
                try config.stream_params.assign(key, val);
            },
            .cameras => {
                if (trimmed.len == 0) { // end of single camera config
                    cameras_parsed += 1;
                    continue;
                }

                var key = try read_until(trimmed, ':');
                const val = trim(trimmed[key.len..]);

                const leading_hash = hash(key[0..2]);
                if (leading_hash == SectionHashes.new_camera) key = key[2..];

                try config.camera_configs[cameras_parsed].assign(key, val);
            },
        }
    }

    return config;
}
