import getpass
import io
import os
import paramiko
import tarfile
from time import sleep

from config_parser import parse_config

USERNAME = "alecfessler"
SSH_PUB_KEY = "~/.ssh/id_rsa"
CAM_CONF_PATH = "/etc/mocap-toolkit/cams.yaml"
CAM_SOFTWARE_PATH = "~/Documents/mocap-toolkit/picam"
REMOTE_SERVICE_PATH = "/etc/systemd/system/"
REMOTE_SERVICE_NAME = "picam.service"
REMOTE_TAR_NAME = "temp.tar.gz"

def run_sudo_cmd(ssh, cmd, pswd):
  channel = ssh.get_transport().open_session()
  channel.get_pty()
  channel.exec_command(f"sudo -s {cmd}")

  print(cmd)

  print(channel.recv(1024).decode().strip())

def main():
  config = parse_config(CAM_CONF_PATH)
  key_path = os.path.expanduser(SSH_PUB_KEY)
  cam_software_path = os.path.expanduser(CAM_SOFTWARE_PATH)

  pswd = getpass.getpass("Enter the sudo password for remote systems: ")

  tar_stream = io.BytesIO()
  with tarfile.open(fileobj=tar_stream, mode="w:gz") as tar:
    tar.add(cam_software_path, arcname=os.path.basename(cam_software_path))
  tar_data = tar_stream.getvalue()

  for camera in config["cameras"]:
    ip = camera["wifi_ip"]

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, username=USERNAME, key_filename=key_path)

    sftp = ssh.open_sftp()

    with sftp.open(REMOTE_TAR_NAME, "wb") as f:
      f.write(tar_data)

    ssh.exec_command(f"tar xzf {REMOTE_TAR_NAME} && rm {REMOTE_TAR_NAME}")
    run_sudo_cmd(
      ssh,
      f"cd picam && make install",
      pswd
    )
    run_sudo_cmd(
      ssh,
      f"cd picam && cp {REMOTE_SERVICE_NAME} {REMOTE_SERVICE_PATH}{REMOTE_SERVICE_NAME}",
      pswd
    )
    run_sudo_cmd(
      ssh,
      f"systemctl restart {REMOTE_SERVICE_NAME}",
      pswd
    )

if __name__ == "__main__":
  main()
