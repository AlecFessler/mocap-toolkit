import posix_ipc
import subprocess

consumer_ready = posix_ipc.Semaphore("/mocap-toolkit_consumer_ready", posix_ipc.O_CREAT)
stop_streams = posix_ipc.Semaphore("/mocap-toolkit_stop_streams", posix_ipc.O_CREAT)

try:
    server_process = subprocess.Popen(
        ["mocap-toolkit-server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    print(f"Server started with PID: {server_process.pid}")

    for i in range(10):
        consumer_ready.acquire()
        print(f"Got frame set {i}")

    stop_streams.release()

    server_process.wait()
    print("Server process terminated gracefully")

finally:
    # Cleanup
    consumer_ready.unlink()
    stop_streams.unlink()
