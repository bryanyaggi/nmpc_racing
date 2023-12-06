#!/bin/bash

username=ubuntu

docker run \
  -it \
  --rm \
  --net=host \
  --ipc=host \
  --privileged \
  --gpus=all \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=$XAUTHORITY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME/courses/MEEN689-Swami/project:/home/"$username"/project \
  -v /storage:/storage \
  --name project-v1 \
  --user "$username" \
  --workdir /home/"$username" \
  --interactive \
  --detach \
  ros-noetic-nmpc-racing
