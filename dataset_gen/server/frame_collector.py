import asyncio
import logging
import time
from dataclasses import dataclass
from threading import atomic
from typing import List, Dict, Any, Callable, Optional

@dataclass
class Frame:
  has_data: bool = False
  data: bytes | None = None
  camera_name: str = ""

@dataclass
class FrameSet:
  timestamp: int  # Nanoseconds since epoch
  total_cameras: int
  frames_filled: atomic.AtomicInteger
  frames: List[Frame]  # Fixed size list, indexed by camera position

class FrameCollector:
  def __init__(
    self,
    start_timestamp: int,
    camera_configs: List[Dict[str, Any]],
    callback: Optional[Callable[[int, List[Frame]], None]] = None,
    frame_duration: int = 33_333_333, # 33.33ms in ns
    logger: logging.Logger
  ):
    self.start_timestamp = start_timestamp
    self.frame_duration = frame_duration
    self.callback = callback
    self.total_cameras = len(camera_configs)
    self.completed_cameras = set()
    self.logger = logger

    # Atomic counter for camera index assignment
    self.next_index = atomic.AtomicInteger(0)
    self.camera_indices: Dict[str, int] = {}

    # Initialize with two empty frame sets
    self.frame_sets: List[FrameSet] = []
    self._add_frame_set()  # Current
    self._add_frame_set()  # Next

    # Set initial processing timestamp
    self.next_process_time = self.start_timestamp + frame_duration

  def mark_camera_complete(self, camera_name: str) -> None:
    """Mark a camera as having completed its stream"""
    self.completed_cameras.add(camera_name)
    self.logger.debug(f"Camera {camera_name} marked complete. {len(self.completed_cameras)}/{self.total_cameras} cameras complete")

  def _add_frame_set(self) -> None:
    """Add a new empty frame set to the list"""
    timestamp = (self.start_timestamp +
                len(self.frame_sets) * self.frame_duration)

    frames = [Frame() for _ in range(self.total_cameras)]
    frame_set = FrameSet(
      timestamp=timestamp,
      total_cameras=self.total_cameras,
      frames_filled=atomic.AtomicInteger(0),
      frames=frames
    )
    self.frame_sets.append(frame_set)

  def collect_frame(self, timestamp: int, camera_name: str, frame_data: bytes) -> None:
    """
    Callback for receiving frames from cameras.
    Assigns camera indices on first frame from each camera.
    """
    # Assign camera index if not already assigned
    if camera_name not in self.camera_indices:
      self.camera_indices[camera_name] = self.next_index.inc()

    camera_idx = self.camera_indices[camera_name]

    # Find matching frame set
    for frame_set in self.frame_sets:
      if frame_set.timestamp == timestamp:
        frame = frame_set.frames[camera_idx]
        frame.has_data = True
        frame.data = frame_data
        frame.camera_name = camera_name
        frame_set.frames_filled.inc()
        break

  def _process_complete_sets(self) -> None:
    """
    Process frame sets, handle incomplete ones if a later one completes.
    Call callback for complete sets if provided.
    """
    while self.frame_sets:
      frame_set = self.frame_sets[0]

      # Check if this or any later set is complete
      found_complete = False
      complete_idx = None

      for idx, fs in enumerate(self.frame_sets):
        if fs.frames_filled.get() == self.total_cameras:
          found_complete = True
          complete_idx = idx
          break

      if not found_complete:
        break

      # Remove any incomplete sets before the complete one
      if complete_idx > 0:
        incomplete = self.frame_sets[:complete_idx]
        self.logger.warning(
          f"Disposing of {len(incomplete)} incomplete frame sets "
          f"(timestamps: {[fs.timestamp for fs in incomplete]})"
        )
        self.frame_sets = self.frame_sets[complete_idx:]

      # Process the complete set
      complete_set = self.frame_sets.pop(0)
      if self.callback:
        self.callback(complete_set.timestamp, complete_set.frames)

      # Ensure we maintain two future frame sets
      while len(self.frame_sets) < 2:
        self._add_frame_set()

  async def monitor_frames(self) -> None:
    """
    Monitor frame sets and process them one frame duration after capture.
    Uses clock sync to align with camera captures.
    """
    while len(self.completed_cameras) < self.total_cameras:
      # Calculate sleep duration
      current_real = time.clock_gettime(time.CLOCK_REALTIME)
      ns_until_target = self.next_process_time - current_real

      # Sleep until next processing time
      await asyncio.sleep(ns_until_target / 1_000_000_000)  # Convert ns to s

      # Process frame sets
      self._process_complete_sets()

      # Update next processing time
      self.next_process_time += self.frame_duration

    # Process any remaining sets
    self._process_complete_sets()
    self.logger.info("All cameras complete, frame collector exiting")
