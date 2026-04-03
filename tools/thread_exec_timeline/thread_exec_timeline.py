import os
import plotly.express as plt
import pandas as pd

from datetime import datetime
from enum import Enum, auto

SERVER_LOG_PATH = "/var/log/mocap-toolkit/server.log"
CAM_LOG_PATH = "~/Documents/mocap-toolkit/logs/"

class PipelineEvent(Enum):
  CAPTURE_START = 0
  CAPTURE_END = 1
  ENCODE_START = 2
  ENCODE_END = 3
  STREAM_START = 4
  STREAM_END = 5
  RECEIVE_START = 6
  RECEIVE_END = 7
  DECODE_START = 8
  DECODE_END = 9
  FRAMESET_COMPLETE = 10

class PlotEvent(Enum):
  CAPTURE = "Frame Capture"
  ENCODE = "Frame Encode"
  STREAM = "Packet Stream"
  RECEIVE = "Packet Receive"
  DECODE = "Packet Decode"
  FRAMESET = "Frameset Complete"

EVENT_MAP = {
  # camera side
  "Queued capture request": PipelineEvent.CAPTURE_START,
  "Completed capture request": PipelineEvent.CAPTURE_END,
  "Started encoding frame": PipelineEvent.ENCODE_START,
  "Finished encoding frame": PipelineEvent.ENCODE_END,
  "Started streaming packet": PipelineEvent.STREAM_START,
  "Finished streaming packet": PipelineEvent.STREAM_END,
  # server side
  "Received packet header": PipelineEvent.RECEIVE_START,
  "Received full packet": PipelineEvent.RECEIVE_END,
  "Started decoding packet": PipelineEvent.DECODE_START,
  "Finished decoding packet": PipelineEvent.DECODE_END,
  "Received full frameset": PipelineEvent.FRAMESET_COMPLETE
}

def extract_timestamp(line):
  sections = line.split()
  timestamp_str = sections[0] + " " + sections[1]
  return datetime.fromisoformat(timestamp_str.strip('Z'))

def parse_logs():
  CAM_LOG_FULL_PATH = os.path.expanduser(CAM_LOG_PATH)

  # Extract benchmark logs from rpicam01
  cam_benchmark_logs = []
  with open(CAM_LOG_FULL_PATH + "rpicam01.log", "r") as f:
    for line in f:
      if "BENCHMARK" in line:
        cam_benchmark_logs.append(line)

  # Extract benchmark logs from server for rpicam01
  server_benchmark_logs = []
  with open(SERVER_LOG_PATH, "r") as f:
    for line in f:
      if "BENCHMARK" in line and "rpicam01" in line:
        server_benchmark_logs.append(line)
      elif "Received full frameset" in line:
        server_benchmark_logs.append(line)

  # Map logs onto tuples of (timestamp, event_enum) pairs
  all_logs = cam_benchmark_logs + server_benchmark_logs
  event_map_keys = EVENT_MAP.keys()
  timed_events = []
  for line in all_logs:
    timestamp = extract_timestamp(line)

    event_enum = -1
    for key in event_map_keys:
      if key in line:
        event_enum = EVENT_MAP[key]

    timed_event = (timestamp, event_enum)
    timed_events.append(timed_event)

  # Return list of timed event tuples sorted by ascending timestamps
  return sorted(timed_events, key=lambda x: x[0])

def create_timeline_df(timed_events):
  timeline_data = []
  pending_starts = {}

  task_map = {
    1: PlotEvent.CAPTURE.value,
    3: PlotEvent.ENCODE.value,
    5: PlotEvent.STREAM.value,
    7: PlotEvent.RECEIVE.value,
    9: PlotEvent.DECODE.value
  }

  for timestamp, event in timed_events:
    if event == -1:
      continue

    event_num = event.value

    if event_num == PipelineEvent.FRAMESET_COMPLETE.value:
      timeline_data.append({
        "Task": PlotEvent.FRAMESET.value,
        "Start": timestamp,
        "Finish": timestamp + pd.Timedelta(microseconds=100)
      })
    elif event_num % 2 == 0:
      pending_starts[event_num] = timestamp
    else:
      start_time = pending_starts.get(event_num - 1)
      if start_time:
        task = task_map[event_num]
        timeline_data.append({
          "Task": task,
          "Start": start_time,
          "Finish": timestamp
        })
        del pending_starts[event_num - 1]

  return pd.DataFrame(timeline_data)

def main():
  timed_events = parse_logs()
  timeline_df = create_timeline_df(timed_events)

  fig = plt.timeline(timeline_df, x_start="Start", x_end = "Finish", y="Task")
  fig.update_yaxes(autorange="reversed")

  fig.update_xaxes(
    rangeslider=dict(visible=True),
    range=[
      timeline_df['Start'].min(),
      timeline_df['Start'].min() + pd.Timedelta(microseconds=100000)
    ]
  )

  fig.update_layout(
    title="Video Pipeline Timeline",
    xaxis_title="Time",
    yaxis_title="Pipeline Stage",
    height=400
  )

  fig.write_html("pipeline_timeline.html")

if __name__ == "__main__":
  main()
