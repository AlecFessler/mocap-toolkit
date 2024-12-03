import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any

def setup_logger(camera_config: Dict[str, Any] = None):
  """
  Configure root logger with rotating file handler.
  If camera_config is provided, sets up logging for specific camera.
  Otherwise sets up main process logging.

  Args:
      camera_config: Optional camera configuration dictionary
  """
  log_dir = Path("/var/log/multicam")
  log_dir.mkdir(exist_ok=True)

  if camera_config:
    logger_name = f"camstream_{camera_config['name']}"
    log_file = log_dir / f"camstream_{camera_config['name']}.log"
  else:
    logger_name = "multicam_main"
    log_file = log_dir / "multicam_main.log"

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
