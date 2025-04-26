import getpass
import io
import os
import tarfile
import yaml

from fabric import Connection, Config

USERNAME = "alecfessler"

CAM_CONF_PATH = "/etc/mocap-toolkit/cams.yaml"
CAM_SOFTWARE_PATH = os.path.expanduser("~/mocap-toolkit/picam")

SERVICE_PATH = "/etc/systemd/system/"
SERVICE_NAME = "picam.service"

TAR_NAME = "temp.tar.gz"

def parse_config(fpath):
  with open(fpath, 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
  return config

def compress_dir(dirname):
  tar_stream = io.BytesIO()
  with tarfile.open(fileobj=tar_stream, mode="w:gz") as tar:
    tar.add(dirname, arcname=os.path.basename(dirname))
  return tar_stream.getvalue()

def main():
  config = parse_config(CAM_CONF_PATH)
  cam_software_path = os.path.expanduser(CAM_SOFTWARE_PATH)

  payload = compress_dir(cam_software_path)
  pswd = getpass.getpass("Enter the sudo password for remote systems: ")
  ssh_config = Config(overrides={"sudo": {"password": pswd}})

  for camera in config["cameras"]:
    conn = Connection(
      host=camera["wifi_ip"],
      user=USERNAME,
      config=ssh_config
    )

    sftp = conn.sftp()
    with sftp.open(TAR_NAME, "wb") as f:
      f.write(payload)

    conn.run(f"tar zxf {TAR_NAME} && rm {TAR_NAME}")
    conn.sudo("bash -c 'cd picam && make install'")
    conn.sudo(f"cp picam/{SERVICE_NAME} {SERVICE_PATH}{SERVICE_NAME}")
    conn.sudo("systemctl daemon-reload")
    conn.sudo(f"systemctl enable {SERVICE_NAME}")
    conn.sudo(f"systemctl restart {SERVICE_NAME}")

if __name__ == "__main__":
  main()
