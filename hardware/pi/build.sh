#!/bin/bash

g++ capture/*.cpp Logger.cpp -o framecap -pthread -lrt `pkg-config --cflags --libs libcamera` -I/home/afessler/Documents/video_capture

sudo setcap cap_sys_nice+ep ./framecap

g++ stream/*.cpp Logger.cpp -o framestream -pthread -lrt -I/home/afessler/Documents/video_capture

sudo setcap cap_sys_nice+ep ./framestream
