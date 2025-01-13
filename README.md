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

## Installation

## Configuration

## Usage

## References
- https://en.wikipedia.org/wiki/Non-blocking_algorithm
