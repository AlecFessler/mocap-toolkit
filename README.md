# 3D Motion Capture Toolkit

## Overview
A high-performance, distributed motion capture system built in C and C++ for Raspberry Pi 5 hardware. The system achieves microsecond-precise frame synchronization across multiple cameras and includes tools for camera calibration, stereo calibration, and 3D pose reconstruction.

The toolkit consists of two main components:
- A video streaming stack that handles synchronized capture across multiple Raspberry Pi cameras
- A processing toolkit for camera calibration and 3D pose reconstruction

## Key Features
- Microsecond-level frame synchronization (~5-25μs precision)
- Distributed video capture on Raspberry Pi 5 hardware
- Precision Time Protocol (PTP) synchronization over standard network hardware
- Low-latency (~45-50ms) video streaming to centralized server
- Near fully automated calibration tools - user simply moves a chessboard pattern within camera view
- 3D pose reconstruction capabilities (**under development**)

## System Architecture
The system is built around a distributed capture network and a centralized processing server. At every stage of the pipeline, including both the server and toolkit components, threads communicate through a highly optimized [single producer single consumer queue](https://github.com/AlecFessler/libspscq). This design choice ensures the entire system end-to-end is both [lock-free and wait-free](https://en.wikipedia.org/wiki/Non-blocking_algorithm), enabling consistent low-latency performance from capture through display/processing with minimal synchronization overhead.

### Capture Network
- Multiple Raspberry Pi 5 nodes running custom network clock sync aware video capture software included in the repo
- PTP-based time synchronization over standard ethernet network hardware, leveraging the Pi 5 ethernet controller's hardware timestamping capabilities
- Video encoder configured for low-latency streaming
- Cameras independently compute the same deterministic capture schedule based on the server's initial timestamp broadcast
- Systemd service for automated camera software management and reliable operation

### Central Server
- Configurable network topology through config file
- Receives and manages synchronized video streams from all cameras
- Hardware accelerated video decoding using Nvidia's NVCUVID
- Bundles incoming frames based on matching timestamps
- Custom wait-free, zero-copy, FIFO IPC protocol using shared memory for downstream tools

### Processing Toolkit
- **Lens Calibration**: Real-time camera matrix and distortion coefficient computation directly from live video stream
- **Stereo Calibration**: Real-time translation and rotation vector computation for 3D reconstruction for all valid camera pairs in the network directly from multi-camera livestream
- **3D Reconstruction**: 3D pose reconstruction using 2D pose predictions per camera view and calibration parameters (not real-time reconstruction, **under development**)

## Requirements

### Hardware Requirements

#### Camera Nodes
- Raspberry Pi 5 (any RAM configuration)
- Raspberry Pi Camera Module 3
- MicroSD card (8GB+ recommended)

#### Server
- Nvidia GPU supporting NVCUVID
- Ethernet interface

#### Network
- Basic gigabit unmanaged switch is sufficient
- Cat5e or better ethernet cables

### Software Requirements

#### Camera Nodes
- Raspberry Pi OS 12 (64-bit)
- chrony 4.3
- ptp4l 3.1.1
- libcamera 0.3.2
- libavcodec 59.37.100
- libavutil 57.28.100

#### Server
- Linux distribution of choice (tested on Arch Linux)
- Nvidia GPU drivers supporting NVCUVID (tested with 535.146.03)
- CUDA 12.8
- libyaml 0.2.5
- libavcodec 61.27.101
- libavutil 59.51.100

**Note**: The FFmpeg install is a little tricky to get right, see details [here](#server-2).

#### Toolkit
- Linux distribution of choice (tested on Arch Linux)
- CUDA 12.8
- libtorch 2.3.0+cu128
- opencv 4.10.0
- libyaml 0.2.5

### Network Requirements
- All devices (cameras and server) must be on the same local network
- Server and cameras must be assigned IPs on the same subnet (e.g., 192.168.1.X/24)
- Static IP addresses for all devices
- Sufficient bandwidth for video streaming (gigabit recommended)
- Cameras configured for PTP sync with each other and NTP sync with server
- Server configured to serve NTP time to cameras

## Installation

### Camera Nodes

#### Network Time Services
```bash
sudo apt install chrony
sudo apt install linuxptp
```

#### Development Tools and Libs
```bash
sudo apt install build-essential
sudo apt install libavcodec-dev libavutil-dev
sudo apt install libcamera-dev
```

### Server

#### Network Time Service
```bash
sudo dnf install chrony
```

#### CUDA
Install the CUDA 12.8 Toolkit from this link:  
https://developer.nvidia.com/cuda-12-8-0-download-archive

#### FFmpeg (with NVENC/NVDEC support)
```bash
# Install build tools
sudo dnf install -y build-essential pkgconfig git yasm nasm cmake

# Clone NVCodec headers
git clone https://code.videolan.org/videolan/ffmpeg/nv-codec-headers.git
cd nv-codec-headers
make
sudo make install

# Clone FFmpeg
cd ..
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
cd ffmpeg

# Configure FFmpeg
./configure --prefix=/usr/local \
            --enable-nonfree \
            --enable-cuda-nvcc \
            --enable-nvenc \
            --enable-nvdec \
            --enable-cuvid \
            --enable-pic \
            --enable-shared \
            --disable-static \
            --extra-cflags='-I/usr/local/include -I/usr/local/include/ffnvcodec -I/usr/local/cuda/include -fPIC' \
            --extra-ldflags='-L/usr/local/cuda/lib64 -fPIC' \
            --nvccflags='-ccbin /usr/bin/gcc-13'

# Build and install
make -j$(nproc)
sudo make install
```

**Note**: Use GCC 14 for building FFmpeg itself, but GCC 13 specifically for nvccflags when configuring.

**Note for Arch Linux users**: Package names on Arch do not have the `-devel` suffix.

#### LibYaml
```bash
sudo dnf install libyaml-devel
```

### Toolkit

#### LibTorch
Download LibTorch (Linux, C++/Java, CUDA 12.8) from:
https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu128.zip

Extract and install accordingly.

#### OpenCV
```bash
sudo dnf install opencv opencv-devel
```

#### Pose Model
Download the pose model from Huggingface:
https://huggingface.co/facebook/sapiens-pose-1b-torchscript

Save the model to:
`/var/lib/mocap-toolkit/sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2`

## Configuration
The system requires configuration files and services to be set up on both the camera nodes and the server. Before proceeding, ensure you have established static IP addresses for all devices in your network, as these will be needed in the configuration files.

**Important**: The server and all camera nodes must be assigned IPs on the same subnet (e.g., 192.168.1.X/24). If the server is on 192.168.86.X and the cameras are on 192.168.1.X, they will not be able to communicate correctly.

### Camera Nodes
The camera nodes require several components to be configured: the time synchronization services, the recording service, and the camera configuration file.

#### Time Synchronization Setup
After installing chrony and linuxptp as described in the [Installation section](#installation), run the provided setup script:

```bash
sudo ./setup_time.sh
```

This script configures both NTP synchronization with the server and PTP synchronization between the cameras. It creates and enables the necessary systemd services for reliable operation.

#### Recording Service Setup
Create the picam service file at `/etc/systemd/system/picam.service`:

```bash
[Unit]
Description=Frame Capture Service
After=network.target
Wants=network.target

[Service]
ExecStart=/usr/local/bin/picam
WorkingDirectory=/usr/local/bin
Restart=always
RestartSec=1
KillSignal=SIGINT
User=$USER
Environment=PATH=/usr/bin:/usr/local/bin

[Install]
WantedBy=multi-user.target
```

After creating the service file, enable and start it:

```bash
sudo systemctl enable picam
sudo systemctl start picam
```

#### Camera Configuration
First, create the necessary directory with appropriate permissions:

```bash
sudo mkdir -p /etc/picam
sudo mkdir -p /var/log/picam
sudo chown $USER:$USER /etc/picam /var/log/picam
```

Create the camera configuration file at `/etc/picam/cam_config.txt`:

```bash
FRAME_WIDTH=1280
FRAME_HEIGHT=720
FPS=30
SERVER_IP=192.168.1.100  # Replace with your server's static IP
TCP_PORT=12345            # Must match port in server config
UDP_PORT=22345            # Must match port in server config
```

**Note**: Changing the resolution and fps from 720p30 is not recommended, but should work within some reasonable range. You need to change this in the server config as well if doing so.

### Server

#### NTP Server Setup
Add the following line to `/etc/chrony.conf` to allow NTP client access from cameras:

```bash
# Allow NTP client access from local network
allow 192.168.1.0/24  # Adjust subnet to match your network
```

Restart the chrony service to apply changes:

```bash
sudo systemctl restart chronyd
```

#### Server Configuration
Create the necessary directories with appropriate permissions:

```bash
sudo mkdir -p /etc/mocap-toolkit
sudo mkdir -p /var/log/mocap-toolkit
sudo chown $USER:$USER /etc/mocap-toolkit /var/log/mocap-toolkit
```

Create the server configuration file at `/etc/mocap-toolkit/cams.yaml`:

```yaml
stream_params:
  frame_width: 1280   # Must match camera settings
  frame_height: 720   # Must match camera settings
  fps: 30             # Must match camera settings

cameras:
  - name: rpicam01          # Must follow this naming convention (rpicamXX)
    id: 0                   # Must increment by 1 for each camera
    eth_ip: 192.168.1.101   # Camera's ethernet IP
    wifi_ip: 192.168.86.106 # Camera's wifi IP (if used, otherwise put a placeholder IP)
    tcp_port: 12345         # Must be unique
    udp_port: 22345         # Must be unique
```

## Usage

The toolkit provides a sequence of tools that must be used in a specific order to set up and operate the motion capture system. This guide will walk you through the process from initial calibration pattern generation through system operation.

### Starting the Frame Server

The frame server must be built and installed before anything else, but execution is handled by whatever tool is currently running. The server handles communication with the cameras, and their service handles their execution ensuring they're always ready for the servers start signal.

```bash
cd mocap-toolkit/frameset_server/
make
sudo make install
```

### Calibration Pattern Generation

Before calibrating your cameras, you'll need to generate a chessboard calibration pattern:

```bash
cd mocap-toolkit/toolkit/
python3 gen_chessboard_pattern.py
```

This will generate a pattern that you can print and mount on a rigid backing like cardboard. The pattern should be kept flat for accurate calibration.

### Camera Calibration Process

The system requires two types of calibration: individual lens calibration for each camera, and stereo calibration to establish spatial relationships between cameras.

#### Lens Calibration

Each camera must first be calibrated individually to determine its intrinsic parameters. To build and run the lens calibration tool:

```bash
cd mocap-toolkit/toolkit/lens_calibration/
make
cd bin
./lens_calibration [camera_id]
```

Move the chessboard pattern through different positions and orientations in front of the camera. The tool will automatically capture calibration data and exit when sufficient samples have been collected. The process generates a .yaml file containing the camera matrix and distortion coefficients. This takes about 3-4 seconds to complete.

This process must be repeated for each camera in your setup, using the appropriate camera_id for each.

#### Stereo Calibration

After completing lens calibration for all cameras, you can establish the geometric relationships between them using the stereo calibration tool:

```bash
cd mocap-toolkit/toolkit/stereo_calibration/
make
cd bin
./stereo_calibration
```

When running stereo calibration, move the chessboard pattern through positions visible to multiple cameras simultaneously. The tool will automatically identify valid camera pairs and compute their relative positions and orientations. For each identified camera pair, the tool generates a .yaml file containing the translation and rotation vectors needed for 3D reconstruction. The duration for this depends on how many cameras you have and their particular setup in the scene, but should only take several seconds to complete.

### 3D Reconstruction Tool

The 3D reconstruction tool for generating motion capture data is currently under development. You can find the work-in-progress implementation in the `toolkit/mocap_dataset_gen/` directory.
