import asyncio
import logging
from stream_ctl import setup_stream_session, load_camera_config, broadcast_stop, setup_logger

async def process_frames(collector, logger):
  """Task to process frame sets as they arrive"""
  try:
    while True:
      frame_set = await collector.frame_queue.get()
      logger.info(f"Received frame set with {frame_set.frames_filled} frames")
      # Add lens calibration here

  except asyncio.CancelledError:
    logger.info("Frame processing cancelled")
    raise

async def main():
  cam_confs = load_camera_config("../../cams.yaml")

  # filter out all but one camera for a single stream

  logger = setup_logger('/var/log/lens_calib', 'lens.log')

  collector, tasks = await setup_stream_session(cam_confs, logger)

  process_task = asyncio.create_task(process_frames(collector, logger))
  tasks.append(process_task)

  done = pending = None
  try:
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    for task in done:
      if task.exception():
        raise task.exception()

  except Exception as e:
    logger.error(f"Error during streaming: {e}")

  finally:
    broadcast_stop(cam_confs, logger)

    if pending:
      for task in pending:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
  asyncio.run(main())
