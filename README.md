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
The system is built around a distributed capture network and a centralized processing server. At every stage of the pipeline, including both the server and toolkit components, threads communicate through a highly optimized [single producer single consumer queue](https://github.com/AlecFessler/libspscq). This design choice ensures the entire system is both [lock-free and wait-free](https://en.wikipedia.org/wiki/Non-blocking_algorithm), enabling consistent low-latency performance from capture through display/processing with minimal synchronization overhead.

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
- Wait-free, zero-copy, FIFO IPC protocol using shared memory for downstream tools

### Processing Toolkit
- **Lens Calibration**: Real-time camera matrix and distortion coefficient computation in directly from live video stream
- **Stereo Calibration**: Real-time translation and rotation vector computation for 3D reconstruction for all valid camera pairs in the network directly from multi-camera livestream
- **3D Reconstruction**: 3D pose reconstruction using 2D pose predictions per camera view and calibration parameters (not real-time reconstruction, **under development**)

## Requirements

### Hardware Requirements

#### Camera Nodes
- Raspberry pi 5 (any RAM configuration)
- Raspberry pi Camera Module 3
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
- Required packages:
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

**Note**: The FFmpeg install is a little tricky to get right, I had to compile from source using specifically gcc 14 with the following configuration flags:
```bash
configuration: --prefix=/usr/local --enable-nonfree --enable-cuda-nvcc --enable-nvenc --enable-nvdec --enable-cuvid --enable-pic --enable-shared --disable-static --extra-cflags='-I/usr/local/include -I/usr/local/include/ffnvcodec -I/usr/local/cuda/include -fPIC' --extra-ldflags='-L/usr/local/cuda/lib64 -fPIC' --nvccflags='-ccbin /usr/local/gcc-12.3.0/bin/g++-12.3'
```

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

## Configuration

## Usage

## References
- https://en.wikipedia.org/wiki/Non-blocking_algorithm
