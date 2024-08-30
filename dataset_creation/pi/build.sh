#!/bin/bash

g++ src/*.cpp  -o framecap -pthread -lrt `pkg-config --cflags --libs libcamera` -I/home/afessler/Documents/video_capture/include

sudo setcap cap_sys_nice+ep ./framecap
