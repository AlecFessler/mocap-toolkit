import numpy as np
import mmap
import posix_ipc
import signal
import subprocess

SERVER_EXE = "mocap-toolkit-server"
SEM_NAME = "/mocap-toolkit_consumer_ready"
SHM_NAME = "/mocap-toolkit_frameset"

class StreamControl:
  def __init__(
    self,
    frame_width,
    frame_height,
    num_cameras
  ):
    try:
      self._server_process = None
      self._shm = None

      self.num_cameras = num_cameras
      self.frame_width = frame_width
      self.frame_height = frame_height
      self.y_size = frame_width * frame_height
      self.uv_size = frame_width * frame_height // 2
      self.frame_size = self.y_size + self.uv_size
      self.frameset_size = self.frame_size * num_cameras

      self._server_process = subprocess.Popen(
        [SERVER_EXE],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
      )

      self._sem = posix_ipc.Semaphore(
        SEM_NAME,
        posix_ipc.O_CREAT
      )

      shm = posix_ipc.SharedMemory(
        SHM_NAME,
        posix_ipc.O_CREAT,
        size=self.frameset_size
      )
      self._shm = mmap.mmap(
        shm.fd,
        shm.size,
        mmap.MAP_SHARED,
        mmap.PROT_READ
      )
      shm.close_fd()

    except Exception as e:
      self.__del__()
      raise RuntimeError(f"Failed to initialize StreamControl: {e}")

  def __del__(self):
    if self._server_process:
      self._server_process.send_signal(signal.SIGTERM)
    if self._shm:
      self._shm.close()

  def framesets_iter(self):
    try:
      while True:
        self._sem.acquire()

        frameset = []

        for cam_idx in range(self.num_cameras):
          offset = cam_idx * self.frame_size
          y_plane = np.frombuffer(
            self._shm[offset:offset + self.y_size],
            dtype=np.uint8
          ).reshape(self.frame_height, self.frame_width)

          offset = offset + self.y_size
          uv_planes = np.frombuffer(
            self._shm[offset:offset + self.uv_size],
            dtype=np.uint8
          ).reshape(self.frame_height//2, self.frame_width)

          frame = {
            'Y': y_plane.copy(),
            'UV': uv_planes.copy()
          }

          frameset.append(frame)

        yield frameset

    except Exception as e:
      print(f"Error in framesets_iter: {e}")
      raise
