#!/bin/bash

docker build \
  --no-cache \
  -f Dockerfile \
  -t "ros-noetic-nmpc-racing" \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) .
