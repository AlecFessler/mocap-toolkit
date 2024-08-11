#!/bin/bash

g++ -pthread -lrt frame_capture.cpp -o frame_capture

sudo setcap cap_sys_nice+ep ./frame_capture
