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
- Cameras indepenently compute the same deterministic capture schedule based on the server's initial timestamp broadcast
- Systemd service for automated camera software management and reliable operation

### Central Server
- Configurable network topology through config file
- Receives and manages synchronized video streams from all cameras
- Hardware accelerated video decoding using Nvidia's NVCUVID
- Bundles incoming frames based on matching timestamps
- Custom Wait-free, zero-copy, FIFO IPC protocol using shared memory for downstream tools

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
- Linux distribution of choice (tested on Fedora 41)
- Nvidia GPU drivers supporting NVCUVID (tested with 565.57.01)
- CUDA 12.3
- libyaml 0.2.5
- libavcodec 61.27.101
- libavutil 59.51.100

**Note**: The FFmpeg install is a little tricky to get right, see details [here](https://github.com/AlecFessler/mocap-toolkit?tab=readme-ov-file#server-2)

#### Toolkit
- Linux distribution of choice (tested on Fedora 41)
- CUDA 12.3
- libtorch 2.5.1+cu121
- opencv 4.10.0
- libyaml 0.2.5

**Note**: There is no libtorch for cuda 12.3 like we're using, but the version compiled for 12.1 is compatible based on my testing

### Network Requirements
- All devices (cameras and server) must be on the same local network
- All devices (cameras and server) should have an established static IP address
- Sufficient bandwidth for video streaming (gigabit recommended)
- Cameras should be configured for PTP sync with each other and NTP sync with server
- Server should be configured to serve NTP time to cameras

## Installation

### Camera Nodes

#### Network Time Services

```bash
sudo apt install chrony
sudo apt install linuxptp
```

**Note**: There is a shell script in the repo that automates setup on the cameras for the network time services after installation, see details in the Configuration section of the Readme

#### Development Tools and Libs

```bash
sudo apt install build-essential
sudo apt install libavcodec-dev libavutil-dev
sudo apt install libcamera-dev
```

**Note**: This step is only necessary if you're building the recording software from source using the provided Makefile

### Server

**Note**: The installation guide is Fedora-based and assumes a fresh install, some details may differ for your machine

#### Network Time Service
```bash
sudo dnf install chrony
```

**Note**: Details on the time service configuration are shown [here](https://github.com/AlecFessler/mocap-toolkit?tab=readme-ov-file#server-3)

#### CUDA
Install the 12.3 CUDA Toolkit from this link: https://developer.nvidia.com/cuda-12-3-0-download-archive

#### Building FFmpeg from source

```bash
# Install build tools
sudo dnf install -y build-essential pkg-config git yasm nasm cmake

# Download the Nvidia Codec SDK
mkdir ~/ffmpeg_build
cd ~/ffmpeg_build
wget https://developer.nvidia.com/video-sdk-12-0-16-7 -O Video_Codec_SDK.zip
unzip Video_Codec_SDK.zip

# Copy headers to /usr/local/include/ffnvcodec
sudo cp -r include/ffnvcodec /usr/local/include/

# Clone FFmpeg
cd ~/ffmpeg_build
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
cd ffmpeg

# Configure the build
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
            --nvccflags='-ccbin /usr/local/gcc-12.3.0/bin/g++-12.3'

# Build and install
make -j$(nproc)
sudo make install
```

**Note**: I had to use gcc 14 specifically to get FFmpeg to compile

#### LibYaml
```bash
sudo dnf install libyaml-devel
```

### Toolkit
**Note**: libyaml is a dependency of the toolkit as well as the server, so if you followed along with the [installation for the server](https://github.com/AlecFessler/mocap-toolkit?tab=readme-ov-file#libyaml) you'll already have it installed, otherwise see the instructions above

#### LibTorch
Install LibTorch with Linux, C++/Java, and CUDA 12.1 selected from this link: https://pytorch.org/get-started/locally/

**Note**: There is no LibTorch for cuda 12.3 like we're using, but the version compiled for 12.1 is compatible based on my testing

#### OpenCV
```bash
sudo dnf install opencv opencv-devel
```

## Configuration
The system requires configuration files and services to be setup on both the camera nodes and the server. Before proceeding, ensure you have established static IP addresses for all devices in your network, as these will be needed in the configuration files.

### Camera Nodes
The camera nodes require several components to be configured: the time synchronization services, the recording service, and the camera configuration file.

#### Time Synchronization Setup
After installing chrony and linuxptp as described in the [Installation section](https://github.com/AlecFessler/mocap-toolkit?tab=readme-ov-file#installation), run the provided setup script:

```bash
sudo ./setup_time.sh
```

This script configures both NTP synchronization with the server and PTP synchronization between the cameras. It creates and enables the necessary systemd services for reliable operation.

#### Recording Service Setup
Create the framecap service file at
`/etc/systemd/system/framecap.service`:

```bash
[Unit]
Description=Frame Capture Service
After=network.target
Wants=network.target

[Service]
ExecStart=/usr/local/bin/framecap
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
sudo systemctl enable framecap
sudo systemctl start framecap
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
SERVER_IP=192.168.86.100  # Replace with your server's static IP
TCP_PORT=12347            # Must match port in server config
UDP_PORT=22347            # Must match port in server config
```

**Note**: Changing the resolution and fps from 720p30 is not recommended, but should work within some reasonable range. You need to change this in the server config as well if doing so.

### Server

#### NTP Server Setup
Add the following line to `/etc/chrony.conf` to allow NTP client access from cameras:

```bash
# Allow NTP client access from local network
allow 192.168.86.0/24  # Adjust subnet to match your network
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
    eth_ip: 192.168.1.102   # Camera's ethernet IP
    wifi_ip: 192.168.86.79  # Camera's wifi IP (if used, otherwise put a placeholder IP)
    tcp_port: 12345         # Must be unique
    udp_port: 22345         # Must be unique
  # Add additional cameras as needed
```

## Usage
