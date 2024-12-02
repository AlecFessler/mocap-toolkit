import asyncio
import logging
import struct
import time
import socket
from typing import List, Dict, Any

logging.basicConfig(
  level=logging.DEBUG,
  format='[%(asctime)s] [%(levelname)s] %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S'
)

async def send_udp_message(message: str | int, camera: Dict[str, Any]) -> bool:
  """
  Send a UDP message to a camera, falling back to wifi if ethernet fails.

  Args:
    message: Either "STOP" or a timestamp in nanoseconds
    camera: Camera configuration dictionary

  Returns:
    bool: True if message was sent successfully, False otherwise
  """
  if message == "STOP":
    payload = message.encode()
  else:
    payload = struct.pack('>Q', message)

  try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    loop = asyncio.get_event_loop()
    await loop.sock_sendto(sock, payload, (camera['eth_ip'], int(camera['udp_port'])))
    logging.info(f"Sent message via ethernet to {camera['eth_ip']} on port {camera['udp_port']}")
    sock.close()
    return True
  except Exception as e:
    logging.warning(f"Failed to send via ethernet to {camera['name']}: {str(e)}")
    try:
      sock.close()
    except:
      pass

  try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    loop = asyncio.get_event_loop()
    await loop.sock_sendto(sock, payload, (camera['wifi_ip'], int(camera['udp_port'])))
    logging.info(f"Sent message via wifi to {camera['wifi_ip']} on port {camera['udp_port']}")
    sock.close()
    return True
  except Exception as e:
    logging.error(f"Failed to send via wifi to {camera['name']}: {str(e)}")
    try:
      sock.close()
    except:
      pass
    return False

async def broadcast_timestamp(cameras: List[Dict[str, Any]]) -> None:
  """
  Broadcast timestamp to all cameras concurrently.

  Args:
    cameras: List of camera configurations

  Raises:
    RuntimeError: If unable to send timestamp to any camera
  """
  timestamp = int(time.time_ns()) + 1_000_000_000  # Nanoseconds since epoch + 1 second for broadcast latency
  logging.info(f"Broadcasting start timestamp: {timestamp}")

  results = await asyncio.gather(
    *[send_udp_message(timestamp, camera) for camera in cameras],
    return_exceptions=True
  )

  failed_cameras = [
    camera['name']
    for result, camera in zip(results, cameras)
    if not result
  ]

  if failed_cameras:
    raise RuntimeError(f"Failed to send timestamp to cameras: {failed_cameras}")

async def broadcast_stop(cameras: List[Dict[str, Any]]) -> None:
  """
  Broadcast stop signal to all cameras concurrently.

  Args:
    cameras: List of camera configurations
  """
  logging.info("Broadcasting stop signal")

  results = await asyncio.gather(
    *[send_udp_message("STOP", camera) for camera in cameras],
    return_exceptions=True
  )

  failed_cameras = [
    camera['name']
    for result, camera in zip(results, cameras)
    if not result
  ]

  if failed_cameras:
    logging.error(f"Failed to send STOP to cameras: {failed_cameras}")
