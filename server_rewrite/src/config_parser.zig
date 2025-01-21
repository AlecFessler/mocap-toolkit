const std = @import("std");

const NoDelimiter = error{NoDelimiter};

fn read_until(src: []const u8, dest: []u8, delimiter: u8) NoDelimiter!u64 {
    for (src, 0..) |char, i| {
        dest[i] = char;
        if (char == delimiter) return i + 1;
    }
    return error.NoDelimiter;
}

fn trim(slice: []const u8) []const u8 {
    var start: usize = 0;
    var end: usize = slice.len;

    // Trim start
    while (start < end and (slice[start] == ' ' or slice[start] == '\n')) : (start += 1) {}

    // Trim end
    while (end > start and (slice[end - 1] == ' ' or slice[end - 1] == '\n')) : (end -= 1) {}

    return slice[start..end];
}

fn count_cameras(file_buffer: []const u8) u64 {
    var offset: u64 = 0;
    var temp_buffer: [64]u8 = undefined;

    var camera_count: u64 = 0;
    var in_cameras_list: bool = false;
    const section_name: []const u8 = "cameras:";

    while (offset < file_buffer.len) {
        const len = read_until(file_buffer[offset..file_buffer.len], &temp_buffer, '\n') catch return 0;
        const line = trim(temp_buffer[0..len]);
        offset += len;

        if (line.len >= section_name.len and std.mem.eql(u8, line, section_name)) {
            in_cameras_list = true;
            continue;
        }

        if (in_cameras_list and line.len >= 2 and std.mem.eql(u8, line[0..2], "- ")) {
            camera_count += 1;
        }
    }

    return camera_count;
}

const ParseError = NoDelimiter || std.fs.File.OpenError || std.fs.File.ReadError;

pub fn parse(filepath: []const u8) ParseError!void {
    const file = try std.fs.openFileAbsolute(filepath, .{ .mode = .read_only });

    var file_buffer: [1024]u8 = undefined; // may need to increase if cam_count grows larger, for now, at 3 cameras the file is ~500 bytes
    const bytes_read = try file.readAll(&file_buffer);

    const camera_count = count_cameras(file_buffer[0..bytes_read]);
    std.debug.print("Camera count {}\n", .{camera_count});

    var offset: u64 = 0;
    var temp_buffer: [64]u8 = undefined;

    const ConfigSection = enum { stream_params, cameras, none };
    var section: ConfigSection = .none;

    while (offset < bytes_read) {
        switch (section) {
            .none => {
                // read line
                const len = try read_until(file_buffer[offset..bytes_read], &temp_buffer, '\n');
                const line = trim(temp_buffer[0..len]);
                offset += len;

                // entering stream params section
                if (std.mem.eql(u8, line, "stream_params:")) {
                    section = .stream_params;
                }

                // entering cameras list section
                if (std.mem.eql(u8, line, "cameras:")) {
                    section = .cameras;
                }
            },

            .stream_params => {
                // read key
                var len = try read_until(file_buffer[offset..bytes_read], &temp_buffer, ':');
                const key = trim(temp_buffer[0..len]);

                // check for end of section
                if (temp_buffer[0] == '\n') {
                    section = .none;
                    continue;
                }

                std.debug.print("key {s}\n", .{key});
                offset += len;

                // read val
                len = try read_until(file_buffer[offset..bytes_read], &temp_buffer, '\n');
                const val = trim(temp_buffer[0..len]);
                std.debug.print("val {s}\n", .{val});
                offset += len;
            },

            .cameras => {
                // read key
                var len = try read_until(file_buffer[offset..bytes_read], &temp_buffer, ':');
                var key = trim(temp_buffer[0..len]);

                // check for new camera start
                if (key.len >= 2 and std.mem.eql(u8, key[0..2], "- ")) {
                    std.debug.print("Found next camera\n", .{});
                    key = key[2..];
                }

                std.debug.print("key {s}\n", .{key});
                offset += len;

                // read val
                len = try read_until(file_buffer[offset..bytes_read], &temp_buffer, '\n');
                const val = trim(temp_buffer[0..len]);
                std.debug.print("val {s}\n", .{val});
                offset += len;
            },
        }
    }
}
