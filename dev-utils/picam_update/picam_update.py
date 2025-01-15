import getpass
import io
import os
import tarfile
from fabric import Connection, Config

from config_parser import parse_config

USERNAME = "alecfessler"
SSH_PUB_KEY = "~/.ssh/id_rsa"

CAM_CONF_PATH = "/etc/mocap-toolkit/cams.yaml"
CAM_SOFTWARE_PATH = "~/Documents/mocap-toolkit/picam"

SERVICE_PATH = "/etc/systemd/system/"
SERVICE_NAME = "picam.service"

TAR_NAME = "temp.tar.gz"

def compress_dir(dirname):
  tar_stream = io.BytesIO()
  with tarfile.open(fileobj=tar_stream, mode="w:gz") as tar:
    tar.add(dirname, arcname=os.path.basename(dirname))
  return tar_stream.getvalue()

def main():
  config = parse_config(CAM_CONF_PATH)
  key_path = os.path.expanduser(SSH_PUB_KEY)
  cam_software_path = os.path.expanduser(CAM_SOFTWARE_PATH)

  payload = compress_dir(cam_software_path)
  pswd = getpass.getpass("Enter the sudo password for remote systems: ")
  ssh_config = Config(overrides={"sudo": {"password": pswd}})

  for camera in config["cameras"]:
    conn = Connection(
      host=camera["wifi_ip"],
      user=USERNAME,
      config=ssh_config,
      connect_kwargs={
        "key_filename": key_path
      }
    )

    sftp = conn.sftp()
    with sftp.open(TAR_NAME, "wb") as f:
      f.write(payload)

    conn.run(f"tar zxf {TAR_NAME} && rm {TAR_NAME}")
    conn.sudo("bash -c 'cd picam && make install'")
    conn.sudo(f"cp picam/{SERVICE_NAME} {SERVICE_PATH}{SERVICE_NAME}")
    conn.sudo(f"systemctl daemon-reload")
    conn.sudo(f"systemctl restart {SERVICE_NAME}")

if __name__ == "__main__":
  main()
