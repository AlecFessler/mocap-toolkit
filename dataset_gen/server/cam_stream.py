import asyncio
import logging
import struct
from typing import Dict, Any

FRAME_MARKER = b'NEWFRAME'
END_STREAM = b'EOSTREAM'

class CamStream:
  def __init__(self, camera_config: Dict[str, Any], logger: logging.Logger):
    self.config = camera_config
    self.server = None
    self.cam_connected = False
    self.logger = logger

  async def parse_stream(self, reader: asyncio.StreamReader) -> None:
    buffer = b''
    while True:
      chunk = await asyncio.wait_for(reader.read(32768), timeout=3)  # 32KB
      if not chunk:
        raise ConnectionResetError("Camera disconnected")

      buffer += chunk

      while True:
        frame_pos = buffer.find(FRAME_MARKER)
        if frame_pos == -1:
          break

        if frame_pos > 0:
          buffer = buffer[frame_pos:]

        header_size = len(FRAME_MARKER) + 8
        if len(buffer) < header_size:
          break

        timestamp_bytes = buffer[len(FRAME_MARKER):header_size]
        timestamp = struct.unpack('<Q', timestamp_bytes)[0]

        next_frame_pos = buffer[header_size:].find(FRAME_MARKER)
        eos_pos = buffer[header_size:].find(END_STREAM)

        if eos_pos != -1 and (next_frame_pos == -1 or eos_pos < next_frame_pos):
          frame_data = buffer[header_size:header_size + eos_pos]
          self.logger.debug(f"Got final frame with timestamp {timestamp} from {self.config['name']}")
          self.logger.debug(f"Got end of stream signal from {self.config['name']}")
          return

        if next_frame_pos == -1:
          break

        frame_data = buffer[header_size:header_size + next_frame_pos]
        self.logger.debug(f"Got frame with timestamp {timestamp} from {self.config['name']}")

        buffer = buffer[header_size + next_frame_pos:]

  async def manage(self) -> None:
    retry_count = 0

    while True:
      try:
        if not self.server:
          self.server = await asyncio.start_server(
            self.handle_connection,
            host='',
            port=int(self.config['tcp_port'])
          )

        async with self.server:
          await self.server.wait_closed()
          return

      except ConnectionResetError:
        self.logger.warning(f"Camera {self.config['name']} disconnected, waiting 3s")
        self.cam_connected = False
        await asyncio.sleep(3)
        if not self.cam_connected:
          raise RuntimeError(f"Camera {self.config['name']} failed to reconnect after disconnect")
        continue

  async def handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    addr = writer.get_extra_info('peername')
    self.logger.info(f"Camera {self.config['name']} connected from {addr}")
    self.cam_connected = True

    try:
      await self.parse_stream(reader)
      self.logger.debug(f"Stream completed for {self.config['name']}")
      if self.server:
        self.server.close()
    except asyncio.TimeoutError:
      self.logger.warning(f"Camera {self.config['name']} timed out - no data received for 3s")
      if self.server:
        self.server.close()
    finally:
      writer.close()
      await writer.wait_closed()
      self.cam_connected = False
