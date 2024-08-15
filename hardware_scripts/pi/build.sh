#!/bin/bash

g++ *.cpp -o frame_capture -pthread -lrt `pkg-config --cflags --libs libcamera`

sudo setcap cap_sys_nice+ep ./frame_capture
