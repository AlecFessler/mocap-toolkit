import yaml

def parse_conf(fpath = "/etc/mocap-toolkit/cams.yaml"):
  with open(fpath, 'r') as file:
    conf = yaml.safe_load(file)

  frame_width = conf['stream_params']['frame_width']
  frame_height = conf['stream_params']['frame_height']
  fps = conf['stream_params']['fps']

  cameras = []
  for camera in conf['cameras']:
    name = camera['name']
    cameras.append(name)

  return frame_width, frame_height, fps, cameras
