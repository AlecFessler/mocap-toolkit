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
  total_cameras: int
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

    self.next_index = 0 # Assigns unique index to cameras
    self.camera_indices: Dict[str, int] = {}

    self.frame_sets: Deque[FrameSet] = deque()
    self._add_frame_set()

  def mark_camera_complete(self, camera_name: str) -> None:
    self.completed_cameras.add(camera_name)
    self.logger.info(
        f"Camera {camera_name} marked complete. "
        f"{len(self.completed_cameras)}/{self.total_cameras} cameras complete"
    )

  def _add_frame_set(self) -> None:
    timestamp = (self.start_timestamp + self.frame_count * self.frame_duration)
    self.frame_count += 1

    frames = [Frame() for _ in range(self.total_cameras)]
    frame_set = FrameSet(
        timestamp=timestamp,
        total_cameras=self.total_cameras,
        frames_filled=0,
        frames=frames
    )
    self.frame_sets.append(frame_set)
    self.logger.debug(f"Created new frame set for timestamp {timestamp}")

  def collect_frame(self, timestamp: int, camera_name: str, frame_data: bytes) -> None:
    if camera_name not in self.camera_indices:
      self.camera_indices[camera_name] = self.next_index
      self.next_index += 1
      self.logger.info(f"Assigned index {self.camera_indices[camera_name]} to camera {camera_name}")

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

  def _process_complete_sets(self) -> None:
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

  async def monitor_frames(self) -> None:
    # wait for all cameras to be marked done
    while len(self.completed_cameras) < self.total_cameras:
      self._process_complete_sets()
      await asyncio.sleep(self.frame_duration / 1_000_000_000)

    # process any remaining sets
    self._process_complete_sets()

    completion_rate = (
        self.complete_sets / (self.complete_sets + self.incomplete_sets)
        if self.complete_sets + self.incomplete_sets > 0
        else 0
    )
    self.logger.info(
        "Frame collector exiting. Final statistics:\n"
        f"Complete sets: {self.complete_sets}\n"
        f"Incomplete sets: {self.incomplete_sets}\n"
        f"Completion rate: {completion_rate:.2%}"
    )
