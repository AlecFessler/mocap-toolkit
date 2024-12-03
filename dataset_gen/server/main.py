import argparse
import asyncio
import logging
from cam_stream import CamStream
from config import load_camera_config
from logging_config import setup_logger
from network import broadcast_stop, broadcast_timestamp

logger = setup_logger()

CAM_CONF_FILE = 'cams.yaml'

async def start_recording(cam_confs):
  """Start recording process"""
  try:
    stream_managers = [
      CamStream(conf)
      for conf in cam_confs
    ]

    timestamp = broadcast_timestamp(cam_confs, logger)

    stream_tasks = [
      asyncio.create_task(manager.manage())
      for manager in stream_managers
    ]

    await asyncio.gather(*stream_tasks)
    return 0

  except Exception as e:
    logger.error(f"Error during recording: {e}")
    broadcast_stop(cam_confs, logger)
    return 1


async def main():
  parser = argparse.ArgumentParser(description='Manage multi-camera recording')
  parser.add_argument('action', choices=['start', 'stop'], help='Action to perform')
  args = parser.parse_args()

  cam_confs = load_camera_config(CAM_CONF_FILE)

  if args.action == 'start':
    return await start_recording(cam_confs)
  if args.action == 'stop':
    broadcast_stop(cam_confs, logger)

  return 0

if __name__ == "__main__":
  exit_code = asyncio.run(main())
  exit(exit_code)
