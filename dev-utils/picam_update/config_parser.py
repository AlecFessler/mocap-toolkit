import yaml

def parse_config(fpath: str):
  with open(fpath, 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
  return config
