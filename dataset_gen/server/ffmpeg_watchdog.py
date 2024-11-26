# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import asyncio
import logging
import subprocess
import psutil
from asyncio import timeout
from pathlib import Path
from typing import Set
import aioinotify
from aioinotify import Inotify, Mask

# Configuration
PORTS = [12345, 12346, 12347]
RECONNECT_TIMEOUT = 1.0  # seconds to wait for potential reconnect
FULL_DISCONNECT_TIMEOUT = 3.0  # seconds to wait for all ports to disconnect
VIDEO_BASE_PATH = "/home/alecfessler/Documents/mimic/data"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('/var/log/ffmpeg_watchdog.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def check_port_listening(port: int) -> bool:
    """Check if FFmpeg is listening on the port without attempting connection."""
    connections = psutil.net_connections()
    return any(conn.laddr.port == port and conn.status == 'LISTEN' 
              for conn in connections)

async def socket_disconnected(port: int) -> None:
    """Block until FFmpeg stops listening on the port."""
    logger.info(f"Waiting for disconnect on port {port}")
    while await check_port_listening(port):
        await asyncio.sleep(0.1)
    logger.info(f"Port {port} disconnected")

async def socket_reconnected(port: int) -> None:
    """Block until timeout or FFmpeg starts listening again."""
    while True:
        if await check_port_listening(port):
            return True
        await asyncio.sleep(0.1)

def get_latest_segment(port: int) -> Path:
    """Get the latest segment file for a given port."""
    base_path = Path(VIDEO_BASE_PATH)
    counter = 0
    with open(".video_counter.txt", "r") as f:
        counter = int(f.read().strip())

    pattern = f"port{port}_vid{counter}_*.mp4"
    segments = sorted(base_path.glob(pattern))
    return segments[-1] if segments else None

async def file_closed(port: int) -> None:
    """Wait for the latest segment file to be closed."""
    latest_segment = get_latest_segment(port)
    if not latest_segment:
        raise RuntimeError(f"No segments found for port {port}")

    inotify = Inotify()
    await inotify.register(latest_segment, Mask.CLOSE_WRITE)

    async for event in inotify:
        if event.mask & Mask.CLOSE_WRITE:
            break

    await inotify.close()

def check_new_segment(port: int) -> bool:
    """Check if a new segment has appeared after what we thought was the last one."""
    current_latest = get_latest_segment(port)
    if not current_latest:
        return False

    # Compare with our previously known latest
    previous_latest = getattr(check_new_segment, f'last_segment_{port}', None)
    setattr(check_new_segment, f'last_segment_{port}', current_latest)

    return previous_latest and current_latest != previous_latest

async def wait_for_all_disconnects(ports: list[int]) -> bool:
    """Wait for all ports to cleanly disconnect."""
    disconnected_ports: Set[int] = set()

    async def wait_for_disconnect(port: int) -> None:
        while True:
            await socket_disconnected(port)

            try:
                async with timeout(RECONNECT_TIMEOUT):
                    if await socket_reconnected(port):
                        logger.info(f"Port {port} reconnected, continuing to monitor")
                        continue
            except TimeoutError:
                pass

            # No reconnect occurred - clean disconnect
            logger.info(f"Clean disconnect confirmed on port {port}")
            disconnected_ports.add(port)
            if len(disconnected_ports) == 1:
                logger.info(f"First disconnect(s) detected on ports: {disconnected_ports}")
            return

    # Start monitoring all ports
    tasks = [wait_for_disconnect(port) for port in ports]

    # Wait for first disconnect
    done, pending = await asyncio.wait(
        tasks, 
        return_when=asyncio.FIRST_COMPLETED
    )

    # Give remaining ports time to disconnect
    try:
        async with timeout(FULL_DISCONNECT_TIMEOUT):
            await asyncio.gather(*pending)
        logger.info(f"All ports disconnected. Disconnect order: {list(disconnected_ports)}")
        return True
    except TimeoutError:
        still_connected = [p for p in ports if p not in disconnected_ports]
        logger.error(f"Early disconnects on ports: {list(disconnected_ports)}")
        logger.error(f"Failed to disconnect: {still_connected}")
        return False

async def wait_for_file_finalization(ports: list[int]) -> None:
    """Wait for all ports to finalize their video files."""
    async def wait_for_port_files(port: int) -> None:
        logger.info(f"Waiting for file finalization on port {port}")
        await file_closed(port)
        # Wait to ensure no new segment
        await asyncio.sleep(1)
        if check_new_segment(port):
            logger.error(f"New segment detected after finalization on port {port}")
            raise RuntimeError(f"Unexpected new segment on port {port}")
        logger.info(f"Files finalized for port {port}")

    await asyncio.gather(*(wait_for_port_files(port) for port in ports))
    logger.info("All files finalized successfully")

async def main():
    """Main entry point for the watchdog."""
    try:
        # Wait for all ports to disconnect
        if not await wait_for_all_disconnects(PORTS):
            logger.error("Failed to get clean disconnect on all ports")
            subprocess.run(["./ffmpeg_mgr.sh", "stop"])
            return

        # Wait for file finalization
        await wait_for_file_finalization(PORTS)

        # All good - restart manager and start preprocessing
        logger.info("Clean shutdown detected, restarting manager and starting preprocessing")
        subprocess.run(["./ffmpeg_mgr.sh", "restart"])
        subprocess.run(["python3", "preprocess.py"])

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        subprocess.run(["./ffmpeg_mgr.sh", "stop"])

if __name__ == "__main__":
    asyncio.run(main())
