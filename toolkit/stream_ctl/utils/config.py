import yaml
from typing import List, Dict, Any

def load_camera_config(config_path: str) -> List[Dict[str, Any]]:
  """
  Load camera configuration from YAML file.

  Args:
    config_path: Path to YAML configuration file

  Returns:
    List of dictionaries containing camera configurations

  Raises:
    FileNotFoundError: If config file doesn't exist
    yaml.YAMLError: If config file is invalid YAML
    KeyError: If required fields are missing
  """
  with open(config_path, 'r') as f:
    try:
      config = yaml.safe_load(f)

      if not isinstance(config, dict) or 'cameras' not in config:
        raise KeyError('Config file must contain a "cameras" list')

      required_fields = {'name', 'id', 'eth_ip', 'wifi_ip', 'tcp_port', 'udp_port'}
      for camera in config['cameras']:
        missing_fields = required_fields - set(camera.keys())
        if missing_fields:
          raise KeyError(f'Camera config missing required fields: {missing_fields}')

      return config['cameras']

    except yaml.YAMLError as e:
      print(f'Error parsing YAML file: {e}')
      raise
