from common.parse_conf import parse_conf
from common.stream_ctl import StreamControl

CONF_PATH = "/etc/mocap-toolkit/cams.yaml"

def main():
  try:
    frame_width, frame_height, fps, cameras = parse_conf(CONF_PATH)
    stream_ctl = StreamControl(
      frame_width,
      frame_height,
      len(cameras)
    )

    for frameset in stream_ctl.framesets_iter():
      print("Received new frameset")

  except Exception as e:
    print(f"Error during lens calibration: {e}")

if __name__ == "__main__":
  main()
