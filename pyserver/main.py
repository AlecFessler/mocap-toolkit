import argparse
import asyncio
import logging
from cam_stream import CamStream
from config import load_camera_config
from frame_collector import FrameCollector
from logging_config import setup_logger
from network import broadcast_stop, broadcast_timestamp

logger = setup_logger()

CAM_CONF_FILE = 'cams.yaml'

async def check_connections(stream_managers, timeout=5):
  """
  Monitor camera connections and raise an error if any cameras
  haven't connected within the timeout period.
  """
  await asyncio.sleep(timeout)

  # Check each camera's connection status
  unconnected = [
    manager.config['name']
    for manager in stream_managers
    if not manager.cam_connected
  ]

  if unconnected:
    raise RuntimeError(
      f"Cameras failed to connect within {timeout} seconds: {', '.join(unconnected)}"
    )

async def start_recording(cam_confs):
  """Start recording process"""
  try:

    timestamp = broadcast_timestamp(cam_confs, logger)

    frame_collector = FrameCollector(
      start_timestamp=timestamp,
      camera_configs=cam_confs,
      logger=logger
    )

    stream_managers = [
      CamStream(conf, frame_collector, logger)
      for conf in cam_confs
    ]

    # Create tasks for both streaming and connection checking
    stream_tasks = [
      asyncio.create_task(manager.manage())
      for manager in stream_managers
    ]
    connection_check = asyncio.create_task(
      check_connections(stream_managers)
    )
    collector_task = asyncio.create_task(
      frame_collector.monitor_frames()
    )

    # Wait for either all streams to complete or the connection check to fail
    done, pending = await asyncio.wait(
      [*stream_tasks, connection_check, collector_task],
      return_when=asyncio.FIRST_EXCEPTION
    )

    # If we're here due to the connection check completing normally,
    # we can cancel it as all cameras connected successfully
    if connection_check in done and not connection_check.exception():
      logger.debug("All cameras connected successfully")

    # Propagate any exceptions from completed tasks
    for task in done:
      if task.exception():
        raise task.exception()

    # Wait for remaining stream tasks to complete
    if pending:
      await asyncio.gather(*pending)

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
