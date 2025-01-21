//! Parses the mocap-toolkit config yaml file format
//! Not a general purpose yaml parser

const std = @import("std");

const SectionHashes = struct {
    const stream_params = std.hash.Fnv1a_64.hash("stream_params:");
    const cameras = std.hash.Fnv1a_64.hash("cameras:");
    const new_camera = std.hash.Fnv1a_64.hash("- ");
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
        const hash = std.hash.Fnv1a_64.hash(trimmed);

        // look for 'cameras:' section header
        if (hash == SectionHashes.cameras) {
            in_cameras_list = true;
            continue;
        }

        // ensure trimmed len is long enough to check the leading 2 chars hash
        if (trimmed.len < 2 or !in_cameras_list) continue;

        // check if the leading 2 chars indicate a new camera
        const leading_hash = std.hash.Fnv1a_64.hash(trimmed[0..2]);
        if (leading_hash == SectionHashes.new_camera) {
            camera_count += 1;
        }
    }

    return camera_count;
}

const ParseError = NoDelimiter || std.fs.File.OpenError || std.fs.File.ReadError;

/// Parses the camera config yaml file
pub fn parse(filepath: []const u8) ParseError!void {
    const file = try std.fs.openFileAbsolute(filepath, .{ .mode = .read_only });
    var file_buffer: [1024]u8 = undefined; // may need to increase if cam_count grows larger, for now, at 3 cameras the file is ~500 bytes
    const bytes_read = try file.readAll(&file_buffer);

    const camera_count = try count_cameras(file_buffer[0..bytes_read]);
    std.debug.print("Camera count {}\n", .{camera_count});

    const ConfigSection = enum { stream_params, cameras, none };
    var section: ConfigSection = .none;
    var offset: u64 = 0;

    while (offset < bytes_read) {
        const line = try read_until(file_buffer[offset..bytes_read], '\n');
        offset += line.len;
        const trimmed = trim(line);

        switch (section) {
            .none => {
                const hash = std.hash.Fnv1a_64.hash(trimmed);
                section = switch (hash) {
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
                const val = trimmed[key.len..];
                std.debug.print("Key {s}: Val {s}\n", .{ key, val });
            },
            .cameras => {
                if (trimmed.len == 0) { // end of single camera config
                    continue;
                }

                var key = try read_until(trimmed, ':');
                const val = trimmed[key.len..];

                const leading_hash = std.hash.Fnv1a_64.hash(key[0..2]);
                if (leading_hash == SectionHashes.new_camera) key = key[2..];

                std.debug.print("Key {s}: Val {s}\n", .{ key, val });
            },
        }
    }
}
