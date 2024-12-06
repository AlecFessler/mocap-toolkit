import asyncio
import logging
import struct
from collections import deque
from typing import Deque, Tuple, List

FRAME_DELIMITER = b'NEWFRAME'  # Must match server
FRAME_COUNT_SIZE = 4  # Size of uint32 for frame count
FRAME_HEADER_SIZE = 8  # Size of camera ID + frame size headers
BUFFER_SIZE = 32768  # 32KB buffer size

class FrameSetClient:
  """
  Receives frame sets from a Unix domain socket using the same pattern as CamStream.
  Initially just receives and discards data to test connectivity.
  """
  def __init__(self, socket_path: str, logger: logging.Logger):
    self.socket_path = socket_path
    self.logger = logger
    self.connected = False
    self.frame_queue: asyncio.Queue[List[Tuple[int, bytes]]] = asyncio.Queue()

  async def parse_stream(self, reader: asyncio.StreamReader) -> None:
    """
    Parses incoming frame sets using a pre-allocated buffer and zero-copy slicing.
    Protocol structure:
    1. 4-byte uint32: number of frames in set
    2. For each frame:
        - 4-byte uint32: camera ID
        - 4-byte uint32: frame data length
        - N bytes: frame data (N = frame length)
    """
    buffer = bytearray(BUFFER_SIZE)  # Pre-allocate buffer
    view = memoryview(buffer)  # Create memory view for efficient slicing
    current_size = 0

    while True:
      # Read new data into remaining buffer space
      chunk = await asyncio.wait_for(
        reader.read(BUFFER_SIZE - current_size),
        timeout=3
      )
      if not chunk:
        raise ConnectionResetError("Server disconnected")

      # Add new data to buffer
      buffer[current_size:current_size + len(chunk)] = chunk
      current_size += len(chunk)

      # Track position in buffer as we process frame sets
      processed = 0
      while processed + 4 <= current_size:
        frame_count = struct.unpack('>I', view[processed:processed + 4].tobytes())[0]
        next_position = processed + 4  # Start after frame count

        # Pre-calculate size needed for complete frame set
        frame_set_size = 4  # Start with frame count size
        frames_scanned = 0  # Track how many frames we successfully scanned

        # First pass: calculate total size needed and verify we have enough data
        for _ in range(frame_count):
          if next_position + 8 > current_size:
            break

          header = view[next_position:next_position + 8]
          _, frame_size = struct.unpack('>II', header.tobytes())
          frame_set_size += 8 + frame_size
          next_position += 8 + frame_size
          frames_scanned += 1

        # If we couldn't scan all frames or don't have enough data, wait for more
        if frames_scanned < frame_count or processed + frame_set_size > current_size:
          break

        # Second pass: process the entire frame set now that we know it's complete
        current_pos = processed + 4
        frames = []

        for _ in range(frame_count):
          frame_header = view[current_pos:current_pos + 8]
          camera_id, frame_size = struct.unpack('>II', frame_header.tobytes())
          current_pos += 8

          frame_data = view[current_pos:current_pos + frame_size].tobytes()
          current_pos += frame_size
          frames.append((camera_id, frame_data))

        await self.frame_queue.put(frames)
        self.logger.debug(f"Received frame set with {frame_count} frames")
        processed += frame_set_size

      # Shift any remaining unprocessed data to start of buffer
      if processed > 0:
        remaining = current_size - processed
        if remaining > 0:
          buffer[:remaining] = buffer[processed:current_size]
        current_size = remaining

  async def manage(self) -> None:
    while True:
      try:
        reader, writer = await asyncio.open_unix_connection(self.socket_path)
        self.connected = True
        self.logger.info("Connected to frame set server")

        try:
          await self.parse_stream(reader)
        finally:
          writer.close()
          await writer.wait_closed()
          self.connected = False

      except (ConnectionError, asyncio.TimeoutError) as e:
        self.logger.warning(f"Connection error: {e}, retrying in 3s")
        self.connected = False
        await asyncio.sleep(3)
        continue
