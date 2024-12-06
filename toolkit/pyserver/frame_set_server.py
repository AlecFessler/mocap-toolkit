import asyncio
import logging
import struct
import os
from typing import List, Optional

class FrameSetServer:
  def __init__(self, socket_path: str, logger: logging.Logger):
    self.socket_path = socket_path
    self.logger = logger
    self.server = None
    self._writer: Optional[asyncio.StreamWriter] = None

  async def manage(self) -> None:
    try:
      try:
        os.unlink(self.socket_path)
      except FileNotFoundError:
        pass

      self.server = await asyncio.start_unix_server(
        self.handle_client,
        path=self.socket_path
      )
      self.logger.info(f"Frame set server started on {self.socket_path}")

      async with self.server:
        await self.server.wait_closed()

    except Exception as e:
      self.logger.error(f"Server error: {e}")
      raise
    finally:
      if self._writer:
        self._writer.close()
        await self._writer.wait_closed()
      try:
        os.unlink(self.socket_path)
      except FileNotFoundError:
        pass

  async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    if self._writer:
      writer.close()
      await writer.wait_closed()
      return

    self._writer = writer
    self.logger.info("Client connected to frame set server")

    try:
      await self.server.wait_closed()
    except Exception as e:
      self.logger.error(f"Client error: {e}")
    finally:
      self._writer = None
      writer.close()
      await writer.wait_closed()
      self.logger.info("Client disconnected from frame set server")

  def stop(self) -> None:
    if self.server:
      self.server.close()

  async def send_frame_set(self, frames: List[tuple[int, bytes]]) -> None:
    if not self._writer:
      return

    try:
      self._writer.write(struct.pack('>I', len(frames)))

      for camera_id, frame_bytes in frames:
        self._writer.write(struct.pack('>II', camera_id, len(frame_bytes)))
        self._writer.write(frame_bytes)

      await self._writer.drain()

    except Exception as e:
      self.logger.error(f"Error sending frame set: {e}")
      self._writer.close()
      await self._writer.wait_closed()
      self._writer = None
