import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from itertools import islice
from typing import List, Dict, Any, Set, Deque

@dataclass
class Frame:
  camera_name: str = ""
  data: bytes | None = None

@dataclass
class FrameSet:
  timestamp: int  # Nanoseconds since epoch
  frames_filled: int
  frames: List[Frame]  # Fixed size list, indexed by camera position

class FrameCollector:
  def __init__(
      self,
      start_timestamp: int,
      camera_configs: List[Dict[str, Any]],
      logger: logging.Logger,
      frame_duration: int = 33_333_333,  # 33.33ms in ns
  ):
    self.start_timestamp = start_timestamp
    self.frame_duration = frame_duration

    self.total_cameras = len(camera_configs)
    self.completed_cameras: Set[str] = set()

    self.logger = logger

    self.frame_count = 0
    self.complete_sets = 0
    self.incomplete_sets = 0

    self.camera_indices: Dict[str, int] = {}
    for index, cam in enumerate(camera_configs):
      self.camera_indices[cam['name']] = index

    self.frame_sets: Deque[FrameSet] = deque()
    self._add_frame_set()

    self.frame_queue = asyncio.Queue()

  def _add_frame_set(self) -> None:
    timestamp = (self.start_timestamp + self.frame_count * self.frame_duration)
    self.frame_count += 1

    frames = [Frame() for _ in range(self.total_cameras)]
    frame_set = FrameSet(
        timestamp=timestamp,
        frames_filled=0,
        frames=frames
    )
    self.frame_sets.append(frame_set)
    self.logger.debug(f"Created new frame set for timestamp {timestamp}")

  async def collect_frame(self, timestamp: int, camera_name: str, frame_data: bytes) -> None:
    camera_idx = self.camera_indices[camera_name]

    if not self.frame_sets or timestamp > self.frame_sets[-1].timestamp:
      while not self.frame_sets or self.frame_sets[-1].timestamp < timestamp:
        self._add_frame_set()

    frame_set = None
    for fs in reversed(self.frame_sets):
      if fs.timestamp == timestamp:
        frame_set = fs
        break

    if frame_set is None:
      self.logger.warning(
          f"Received frame with timestamp {timestamp} before "
          f"earliest frame set {self.frame_sets[0].timestamp}, dropping frame"
      )
      return

    frame = frame_set.frames[camera_idx]
    frame.data = frame_data
    frame.camera_name = camera_name
    frame_set.frames_filled += 1
    self.logger.debug(
        f"Collected frame from {camera_name} for timestamp {timestamp}. "
        f"Set now has {frame_set.frames_filled}/{self.total_cameras} frames"
    )

    if frame_set.frames_filled == self.total_cameras:
      await self._process_complete_sets()

  async def _process_complete_sets(self) -> None:
    while self.frame_sets:
      frame_set = self.frame_sets[0]
      found_complete = False
      complete_idx = None

      for idx, fs in enumerate(self.frame_sets):
        if fs.frames_filled == self.total_cameras:
          found_complete = True
          complete_idx = idx
          break

      if not found_complete:
        break

      if complete_idx > 0:
        incomplete = list(islice(self.frame_sets, complete_idx))
        self.incomplete_sets += len(incomplete)
        for fs in incomplete:
          self.logger.warning(
              f"Disposing incomplete frame set (timestamp: {fs.timestamp}, "
              f"frames: {fs.frames_filled}/{self.total_cameras})"
          )
        for _ in range(complete_idx):
          self.frame_sets.popleft()

      complete_set = self.frame_sets.popleft()
      self.complete_sets += 1
      self.logger.info(
          f"Processing complete frame set {complete_set.timestamp}. "
          f"Stats: {self.complete_sets} complete, {self.incomplete_sets} incomplete"
      )

      await self.frame_queue.put(complete_set)
