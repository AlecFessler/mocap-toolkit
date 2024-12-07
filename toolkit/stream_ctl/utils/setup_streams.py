import argparse
import asyncio
import logging
import os
import sys
from typing import List, Dict, Any, Tuple

from ..streaming.cam_stream import CamStream
from .config import load_camera_config
from .frame_collector import FrameCollector
from ..streaming.network import broadcast_stop, broadcast_timestamp

CAM_CONF_FILE = '../../cams.yaml'

async def check_connections(stream_managers: List[CamStream], timeout: int = 5) -> None:
  """
  Monitor camera connections and raise an error if any cameras
  haven't connected within the timeout period.
  """
  await asyncio.sleep(timeout)

  unconnected = [
    manager.config['name']
    for manager in stream_managers
    if not manager.cam_connected
  ]

  if unconnected:
    raise RuntimeError(
      f"Cameras failed to connect within {timeout} seconds: {', '.join(unconnected)}"
    )

async def setup_stream_session(
  cam_confs: List[Dict[str, Any]],
  logger: logging.Logger
) -> tuple[FrameCollector, List[asyncio.Task]]:

    timestamp = broadcast_timestamp(cam_confs, logger)

    collector = FrameCollector(
      start_timestamp=timestamp,
      camera_configs=cam_confs,
      logger=logger
    )

    stream_managers = [
      CamStream(conf, collector, logger)
      for conf in cam_confs
    ]

    tasks = [
      asyncio.create_task(manager.manage())
      for manager in stream_managers
    ]

    tasks.append(asyncio.create_task(
      check_connections(stream_managers)
    ))

    return collector, tasks
