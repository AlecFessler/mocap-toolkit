import os
import yaml

from fabric import Connection

USERNAME = "alecfessler"

CAM_CONF_PATH = "/etc/mocap-toolkit/cams.yaml"
CAM_LOG_PATH = "/var/log/picam/picam.log"
OUT_PATH = "~/Documents/mocap-toolkit/logs/"

def parse_config(fpath):
  with open(fpath, 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
  return config

def main():
  config = parse_config(CAM_CONF_PATH)
  out_path = os.path.expanduser(OUT_PATH)

  for camera in config["cameras"]:
    conn = Connection(
      host=camera["wifi_ip"],
      user=USERNAME
    )

    conn.get(CAM_LOG_PATH, f"{out_path}/{camera['name']}.log")

if __name__ == "__main__":
  main()
