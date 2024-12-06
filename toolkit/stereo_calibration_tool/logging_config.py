import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any

def setup_logger():
  """
  Configure root logger with rotating file handler.
  """
  log_dir = Path("/var/log/stereo_calibration")
  log_dir.mkdir(exist_ok=True)

  logger_name = "stereo_calibration"
  log_file = log_dir / "stereo_calibration.log"

  logger = logging.getLogger(logger_name)
  logger.setLevel(logging.DEBUG)

  handler = RotatingFileHandler(
    log_file,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
  )

  formatter = logging.Formatter(
      '[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s] %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S'
  )
  handler.setFormatter(formatter)

  logger.addHandler(handler)

  return logger
