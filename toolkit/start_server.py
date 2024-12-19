import posix_ipc
import signal
import subprocess

consumer_ready = posix_ipc.Semaphore("/mocap-toolkit_consumer_ready", posix_ipc.O_CREAT)

try:
  server_process = subprocess.Popen(
    ["mocap-toolkit-server"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
  )

  print(f"Server started with PID: {server_process.pid}")

  for i in range(1000):
    consumer_ready.acquire()
    print(f"Got frame set {i}")

finally:
  server_process.send_signal(signal.SIGTERM)
  server_process.wait()
  print("Server process terminated gracefully")
