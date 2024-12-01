import socket
import struct
import subprocess
import sys
import time
from datetime import datetime
import argparse

TIMESTAMP_MARKER = 0xFEFEFEFE

def create_ffmpeg_process(output_pattern):
    """Create ffmpeg process for HEVC transcoding with segmented output"""
    cmd = [
        'ffmpeg',
        '-f', 'h264',           # Input format is H.264
        '-i', 'pipe:0',         # Read from stdin
        '-c:v', 'hevc_nvenc',   # NVIDIA HEVC encoder
        '-preset', 'slow',      # Higher quality encoding
        '-crf', '20',           # Constant rate factor (quality)
        '-movflags', '+faststart+frag_keyframe',  # Optimize for streaming
        '-f', 'segment',        # Enable segmented output
        output_pattern          # Output filename pattern
    ]
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def process_stream(client_socket, ffmpeg_proc, log_file=None):
    """Process incoming stream data with timestamp markers"""
    buffer = bytearray()

    while True:
        # Read more data if buffer is low
        if len(buffer) < 4096:
            try:
                data = client_socket.recv(4096)
                if not data:
                    return
                buffer.extend(data)
            except ConnectionError:
                print("Connection lost")
                return

        # Need at least 4 bytes to check for marker
        if len(buffer) < 4:
            continue

        # Check for timestamp marker
        if len(buffer) >= 4 and struct.unpack('!I', buffer[:4])[0] == TIMESTAMP_MARKER:
            # Need full timestamp (4 byte marker + 8 byte timestamp)
            if len(buffer) < 12:
                continue

            # Extract and log timestamp
            timestamp = struct.unpack('!Q', buffer[4:12])[0]
            now = time.time_ns()
            delay_ms = (now - timestamp) / 1_000_000

            if log_file:
                log_file.write(f"{datetime.fromtimestamp(timestamp/1e9).isoformat()},{timestamp},{now},{delay_ms:.2f}\n")
                log_file.flush()

            # Remove marker and timestamp from buffer
            buffer = buffer[12:]
        else:
            # Everything up to the next marker (or end of buffer) is frame data
            # Look for next marker
            next_marker = buffer[4:].find(struct.pack('!I', TIMESTAMP_MARKER))

            if next_marker == -1:
                # No marker found, keep last 4 bytes in case they're start of marker
                if len(buffer) > 4:
                    try:
                        ffmpeg_proc.stdin.write(buffer[:-4])
                        ffmpeg_proc.stdin.flush()
                    except BrokenPipeError:
                        print("ffmpeg process closed pipe")
                        return
                    buffer = buffer[-4:]
            else:
                # Write everything up to the marker
                try:
                    ffmpeg_proc.stdin.write(buffer[:next_marker+4])
                    ffmpeg_proc.stdin.flush()
                except BrokenPipeError:
                    print("ffmpeg process closed pipe")
                    return
                buffer = buffer[next_marker+4:]

def main():
    parser = argparse.ArgumentParser(description='Video stream bridge with timestamp extraction and HEVC transcoding')
    parser.add_argument('--port', type=int, required=True, help='TCP port to listen on')
    parser.addargument('--output', type=str, required=True, help='Output filename pattern (e.g., output%03d.mp4)')
    parser.add_argument('--log', type=str, help='Log file for timestamps (optional)')
    args = parser.parse_args()

    # Create server socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', args.port))
    server.listen(1)

    print(f"Listening on port {args.port}")

    while True:
        # Accept new connection
        client, addr = server.accept()
        print(f"Accepted connection from {addr}")

        # Start ffmpeg
        ffmpeg = create_ffmpeg_process(args.output)

        try:
            # Open log file if specified
            log_file = open(args.log, 'a') if args.log else None

            # Process the stream
            process_stream(client, ffmpeg, log_file)

        except Exception as e:
            print(f"Error processing stream: {e}")
        finally:
            # Clean up
            if log_file:
                log_file.close()
            try:
                ffmpeg.stdin.close()
            except:
                pass
            ffmpeg.terminate()
            ffmpeg.wait()
            client.close()
            print("Connection closed, waiting for new connection")

if name == "main":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
