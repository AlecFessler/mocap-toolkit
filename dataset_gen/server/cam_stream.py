import asyncio
import logging
import socket
import struct
import threading
from typing import Dict, Any

FRAME_MARKER = b'HELLYEAH'

class CamStream:
  def __init__(self, camera_config: Dict[str, Any], stop_flag: threading.Event):
    """
    Initialize camera stream manager.

    Args:
        camera_config: Camera configuration from YAML
        stop_flag: Shared flag indicating clean shutdown
    """
    self.config = camera_config
    self.stop_flag = stop_flag
    self.socket = None

  def connect(self) -> None:
    """Create and bind TCP socket"""
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.socket.bind(('', int(self.config['tcp_port'])))
    self.socket.listen(1)
    self.socket.settimeout(1)

  async def parse_stream(self, conn: socket.socket) -> None:
    """
    Parse incoming TCP stream for markers, timestamps, and frame data.
    Raises any connection errors to manage().

    Format: HELLYEAH | timestamp (8 bytes) | frame data | HELLYEAH | ...
    """
    buffer = b''
    while True:
      chunk = conn.recv(8192)
      if not chunk:
        raise ConnectionResetError("Camera disconnected")

      buffer += chunk
      while True:
        marker_pos = buffer.find(FRAME_MARKER)
        if marker_pos == -1:
          break

        if marker_pos > 0:
          buffer = buffer[marker_pos:]

        if len(buffer) < len(FRAME_MARKER) + 8:
          break

        timestamp_bytes = buffer[len(FRAME_MARKER):len(FRAME_MARKER) + 8]
        timestamp = struct.unpack('>Q', timestamp_bytes)[0]

        next_marker = buffer[len(FRAME_MARKER) + 8:].find(FRAME_MARKER)
        if next_marker == -1:
          break

        frame_data = buffer[len(FRAME_MARKER) + 8:len(FRAME_MARKER) + 8 + next_marker]

        buffer = buffer[len(FRAME_MARKER) + 8 + next_marker:]

        logging.debug(f"Got frame with timestamp {timestamp} from {self.config['name']}")

  async def manage(self) -> None:
    """
    Manage camera connection lifecycle.

    - Waits for initial connection
    - Handles reconnection attempts
    - Raises unrecoverable errors to main
    """
    retry_count = 0

    while not self.stop_flag.is_set():
      try:
        if not self.socket:
          try:
            self.connect()
          except Exception as e:
            retry_count += 1
            if retry_count > 3:
              raise RuntimeError(f"Failed to create socket after 3 attempts: {e}")
            logging.warning(f"Socket creation attempt {retry_count} failed, waiting {retry_count}s")
            await asyncio.sleep(retry_count)
            continue

        while not self.stop_flag.is_set():
          try:
            conn, addr = self.socket.accept()
            conn.settimeout(1)
            logging.info(f"Camera {self.config['name']} connected from {addr}")
            retry_count = 0
            break
          except socket.timeout:
            continue

        if self.stop_flag.is_set():
          return

        try:
          await self.parse_stream(conn)
        except ConnectionResetError:
          if self.stop_flag.is_set():
            return
          logging.warning(f"Camera {self.config['name']} disconnected, waiting 3s")
          await asyncio.sleep(3)

          try:
            self.socket.accept()
            logging.info(f"Camera {self.config['name']} reconnected from {addr}")
            continue
          except socket.timeout:
            raise RuntimeError(f"Camera {self.config['name']} failed to reconnect after disconnect")

        except ConnectionAbortError:
          logging.warning(f"Connection aborted for {self.config['name']}, reconnecting")
          continue
        except socket.timeout:
          raise RuntimeError(f"Timeout waiting for data from {self.config['name']}")
        finally:
          conn.close()

      except (socket.timeout, RuntimeError) as e:
        self.cleanup()
        raise

  def cleanup(self) -> None:
    """Clean up resources"""
    if self.socket:
      self.socket.close()
      self.socket = None
