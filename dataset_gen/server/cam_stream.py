import asyncio
import logging
import struct
import threading
from logging_config import setup_logger
from typing import Dict, Any

FRAME_MARKER = b'NEWFRAME'
END_STREAM = b'EOSTREAM'

class CamStream:
  def __init__(self, camera_config: Dict[str, Any]):
    """
    Initialize camera stream manager.

    Args:
        camera_config: Camera configuration from YAML
    """
    self.config = camera_config
    self.server = None
    self.cam_connected = False
    self.end_of_stream = False
    self.logger = setup_logger(camera_config)

  async def parse_stream(self, reader: asyncio.StreamReader) -> None:
    """
    Parse incoming TCP stream for markers, timestamps, and frame data.
    Handles frames followed by EOSTREAM marker.

    Format:
    - Frames: NEWFRAME | timestamp (8 bytes) | frame data
    - Last frame may be followed by EOSTREAM
    """
    buffer = b''
    while True:
      chunk = await reader.read(65536)  # 64KB
      if not chunk:
        raise ConnectionResetError("Camera disconnected")

      buffer += chunk

      while True:
        # Look for frame marker
        frame_pos = buffer.find(FRAME_MARKER)
        if frame_pos == -1:
          break

        # Clean up any data before the frame marker
        if frame_pos > 0:
          buffer = buffer[frame_pos:]

        # Check if we have enough data for marker + timestamp
        header_size = len(FRAME_MARKER) + 8
        if len(buffer) < header_size:
          break

        # Extract timestamp
        timestamp_bytes = buffer[len(FRAME_MARKER):header_size]
        timestamp = struct.unpack('<Q', timestamp_bytes)[0]

        # Look for next frame marker and EOSTREAM
        next_frame_pos = buffer[header_size:].find(FRAME_MARKER)
        eos_pos = buffer[header_size:].find(END_STREAM)

        # If we found EOSTREAM, use its position as the frame boundary
        if eos_pos != -1 and (next_frame_pos == -1 or eos_pos < next_frame_pos):
          frame_data = buffer[header_size:header_size + eos_pos]
          self.logger.debug(f"Got final frame with timestamp {timestamp} from {self.config['name']}")
          self.logger.debug(f"Got end of stream signal from {self.config['name']}")
          self.end_of_stream = True
          return

        # If no next marker or EOSTREAM, need more data
        if next_frame_pos == -1:
          break

        # Extract and process normal frame
        frame_data = buffer[header_size:header_size + next_frame_pos]
        self.logger.debug(f"Got frame with timestamp {timestamp} from {self.config['name']}")

        # Update buffer
        buffer = buffer[header_size + next_frame_pos:]

  async def handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Handle single client connection"""
    addr = writer.get_extra_info('peername')
    self.logger.info(f"Camera {self.config['name']} connected from {addr}")

    self.cam_connected = True

    try:
      await self.parse_stream(reader)
    finally:
      writer.close()
      await writer.wait_closed()

  async def manage(self) -> None:
    """
    Manage camera connection lifecycle.

    - Waits for initial connection
    - Handles reconnection attempts
    - Raises unrecoverable errors to main
    """
    retry_count = 0
    self.logger.debug(f"Starting manage for camera {self.config['name']}")
    while not self.end_of_stream:
      try:
        if not self.server:
          self.logger.debug(f"Creating server for camera {self.config['name']}")
          try:
            self.server = await asyncio.start_server(
              self.handle_connection,
              host='',
              port=int(self.config['tcp_port'])
            )
          except Exception as e:
            retry_count += 1
            if retry_count > 3:
              raise RuntimeError(f"Failed to create server after 3 attempts: {e}")
            self.logger.warning(f"Server creation attempt {retry_count} failed, waiting {retry_count}s")
            await asyncio.sleep(retry_count)
            continue

        async with self.server:
          await self.server.serve_forever()

      except ConnectionResetError:
        self.logger.warning(f"Camera {self.config['name']} disconnected, waiting 3s")
        self.cam_connected = False
        await asyncio.sleep(3)
        if not self.cam_connected:
          raise RuntimeError(f"Camera {self.config['name']} failed to reconnect after disconnect")
        continue

      except ConnectionAbortedError:
        self.logger.warning(f"Connection aborted for {self.config['name']}, reconnecting")
        self.cam_connected = False
        continue

  def cleanup(self) -> None:
    """Clean up resources"""
    if self.server:
      self.server.close()
