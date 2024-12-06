import asyncio
import logging
from frame_set_client import FrameSetClient
from logging_config import setup_logger

async def main():
    logger = setup_logger()

    # Start server process without trying to monitor it
    process = await asyncio.create_subprocess_exec(
        'python', '../pyserver/main.py', 'start'
    )

    # Brief pause to let server initialize
    await asyncio.sleep(2)

    client = FrameSetClient('/tmp/frame_stream.sock', logger)

    try:
        await client.manage()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await asyncio.create_subprocess_exec(
            'python', '../pyserver/main.py', 'stop'
        )
        process.terminate()
        await process.wait()

    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
