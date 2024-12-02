import argparse
import asyncio
import logging
import os
import signal
import threading
from pathlib import Path
from config import load_camera_config
from network import broadcast_stop, broadcast_timestamp
from cam_stream import CamStream

logging.basicConfig(
  level=logging.DEBUG,
  format='[%(asctime)s] [%(levelname)s] %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S'
)

PID_FILE = '/tmp/multicam_record.pid'
CAM_CONF_FILE = 'cams.yaml'

stop_flag = threading.Event()

def signal_handler(signum, frame):
  """Signal handler for SIGUSR1"""
  logging.info("Received stop signal")
  stop_flag.set()

async def start_recording(cam_confs):
  """Start recording process"""
  if Path(PID_FILE).exists():
    logging.error("Recording process already running")
    return 1

  pid = os.getpid()
  try:
    with open(PID_FILE, 'w') as f:
      f.write(str(pid))
  except Exception as e:
    logging.error(f"Failed to write PID file: {e}")
    return 1

  signal.signal(signal.SIGUSR1, signal_handler)

  try:
    stream_managers = [
      CamStream(conf, stop_flag)
      for conf in cam_confs
    ]

    await broadcast_timestamp(cam_confs)

    stream_tasks = [
      asyncio.create_task(manager.manage())
      for manager in stream_managers
    ]

    await asyncio.gather(*stream_tasks)

  except Exception as e:
    logging.error(f"Error during recording: {e}")
    return 1
  try:
    Path(PID_FILE).unlink()
  except Exception as e:
    logging.error(f"Failed to remove PID file: {e}")

  return 0

async def stop_recording(cam_confs):
  """Stop recording process"""
  try:
    with open(PID_FILE, 'r') as f:
      pid = int(f.read().strip())
  except FileNotFoundError:
    logging.error("No recording process found")
    return 1
  except Exception as e:
    logging.error(f"Error reading PID file: {e}")
    return 1

  try:
    os.kill(pid, signal.SIGUSR1)
  except ProcessLookupError:
    logging.error("Recording process not found")
    Path(PID_FILE).unlink()  # Clean up stale PID file
    return 1
  except Exception as e:
    logging.error(f"Failed to send signal: {e}")
    return 1
  finally:
    await broadcast_stop(cam_confs)

  return 0

async def main():
  parser = argparse.ArgumentParser(description='Manage multi-camera recording')
  parser.add_argument('action', choices=['start', 'stop'], help='Action to perform')
  args = parser.parse_args()

  cam_confs = load_camera_config(CAM_CONF_FILE)

  if args.action == 'start':
    return await start_recording(cam_confs)
  elif args.action == 'stop':
    return await stop_recording(cam_confs)
  else:
    logging.error(f"Invalid action provided: {args.action}")
    return 1

if __name__ == "__main__":
  exit_code = asyncio.run(main())
  exit(exit_code)
