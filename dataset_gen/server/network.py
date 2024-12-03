import logging
import struct
import time
import socket
from typing import List, Dict, Any

def send_udp_message(message: str | int, camera: Dict[str, Any], logger: logging.Logger) -> bool:
  """
  Send a UDP message to a camera, falling back to wifi if ethernet fails.
  Args:
      message: Either "STOP" or a timestamp in nanoseconds
      camera: Camera configuration dictionary
      logger: Logger instance for recording events
  Returns:
      bool: True if message was sent successfully, False otherwise
  """
  if message == "STOP":
    payload = message.encode()
  else:
    payload = struct.pack('>Q', message)

  try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(payload, (camera['eth_ip'], int(camera['udp_port'])))
    logger.info(f"Sent message via ethernet to {camera['eth_ip']} on port {camera['udp_port']}")
    sock.close()
    return True
  except Exception as e:
    logger.warning(f"Failed to send via ethernet to {camera['name']}: {str(e)}")
    try:
      sock.close()
    except:
      pass

  try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(payload, (camera['wifi_ip'], int(camera['udp_port'])))
    logger.info(f"Sent message via wifi to {camera['wifi_ip']} on port {camera['udp_port']}")
    sock.close()
    return True
  except Exception as e:
    logger.error(f"Failed to send via wifi to {camera['name']}: {str(e)}")
    try:
      sock.close()
    except:
      pass
    return False

def broadcast_timestamp(cameras: List[Dict[str, Any]], logger: logging.Logger) -> int:
  """
  Broadcast timestamp to all cameras.
  Args:
      cameras: List of camera configurations
      logger: Logger instance for recording events
  Returns:
      int: The timestamp that was broadcast
  Raises:
      RuntimeError: If unable to send timestamp to any camera
  """
  timestamp = int(time.time_ns()) + 1_000_000_000
  logger.info(f"Broadcasting start timestamp: {timestamp}")

  failed_cameras = []
  for camera in cameras:
    if not send_udp_message(timestamp, camera, logger):
      failed_cameras.append(camera['name'])

  if failed_cameras:
    raise RuntimeError(f"Failed to send timestamp to cameras: {failed_cameras}")

  return timestamp

def broadcast_stop(cameras: List[Dict[str, Any]], logger: logging.Logger) -> None:
  """
  Broadcast stop signal to all cameras.
  Args:
      cameras: List of camera configurations
      logger: Logger instance for recording events
  """
  logger.info("Broadcasting stop signal")

  failed_cameras = []
  for camera in cameras:
    if not send_udp_message("STOP", camera, logger):
      failed_cameras.append(camera['name'])

  if failed_cameras:
    logger.error(f"Failed to send STOP to cameras: {failed_cameras}")
