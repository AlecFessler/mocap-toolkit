#!/bin/bash
trtexec \
  --onnx=/var/lib/mocap-toolkit/sapiens.onnx \
  --saveEngine=/home/alec/mocap-toolkit/models/sapiens.plan \
  --minShapes=input:1x3x1024x768 \
  --optShapes=input:3x3x1024x768 \
  --maxShapes=input:3x3x1024x768 \
  --fp16 \
  --builderOptimizationLevel=5
