#!/bin/bash

g++ *.cpp -o vidcap -pthread -lrt `pkg-config --cflags --libs libcamera`

sudo setcap cap_sys_nice+ep ./vidcap
