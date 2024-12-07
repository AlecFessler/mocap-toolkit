import asyncio
import logging
import struct
from typing import Dict, Any

from ..utils.frame_collector import FrameCollector

FRAME_MARKER = b'NEWFRAME'
END_STREAM = b'EOSTREAM'
BUFFER_SIZE = 32768 # 32KB
HEADER_SIZE = len(FRAME_MARKER) + 8

class CamStream:
  def __init__(self, camera_config: Dict[str, Any], frame_collector: FrameCollector, logger: logging.Logger):
    self.config = camera_config
    self.server = None
    self.cam_connected = False
    self.frame_collector = frame_collector
    self.logger = logger

  async def parse_stream(self, reader: asyncio.StreamReader) -> None:
    buffer = bytearray(BUFFER_SIZE)
    view = memoryview(buffer)
    current_size = 0

    while True:
      chunk = await asyncio.wait_for(
        reader.read(BUFFER_SIZE - current_size),
        timeout=3
      )
      if not chunk:
        raise ConnectionResetError("Camera disconnected")

      buffer[current_size:current_size + len(chunk)] = chunk
      current_size += len(chunk)

      processed = 0
      while processed < current_size:
        remaining = view[processed:current_size]
        frame_pos = remaining.tobytes().find(FRAME_MARKER)

        if frame_pos == -1:
          break

        processed += frame_pos
        if current_size - processed < HEADER_SIZE:
          break

        timestamp_bytes = view[processed + len(FRAME_MARKER):processed + HEADER_SIZE]
        timestamp = struct.unpack('<Q', timestamp_bytes.tobytes())[0]

        data_start = processed + HEADER_SIZE
        remaining = view[data_start:current_size]
        remaining_bytes = remaining.tobytes()
        next_frame_pos = remaining_bytes.find(FRAME_MARKER)
        eos_pos = remaining_bytes.find(END_STREAM)

        if eos_pos != -1 and (next_frame_pos == -1 or eos_pos < next_frame_pos):
          frame_data = remaining_bytes[:eos_pos]
          await self.frame_collector.collect_frame(
            timestamp,
            self.config['name'],
            frame_data
          )
          self.logger.debug(f"Got final frame with timestamp {timestamp} from {self.config['name']}")
          return

        if next_frame_pos == -1:
          break

        frame_data = remaining_bytes[:next_frame_pos]
        await self.frame_collector.collect_frame(
          timestamp,
          self.config['name'],
          frame_data
        )
        processed += HEADER_SIZE + next_frame_pos

      if processed > 0:
        if current_size > processed:
          buffer[:current_size - processed] = buffer[processed:current_size]
        current_size -= processed

  async def manage(self) -> None:
    while True:
      try:
        if not self.server:
          self.server = await asyncio.start_server(
            self.handle_connection,
            host='',
            port=int(self.config['tcp_port'])
          )

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
