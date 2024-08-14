#!/bin/bash

g++ frame_capture.cpp CameraHandler.cpp Logger.cpp -o frame_capture -pthread -lrt `pkg-config --cflags --libs libcamera`

sudo setcap cap_sys_nice+ep ./frame_capture
