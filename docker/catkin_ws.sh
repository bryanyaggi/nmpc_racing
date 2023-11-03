#!/bin/bash

catkin_ws_name=catkin_ws
user=`whoami`

source /opt/ros/noetic/setup.bash
mkdir -p /home/"$user"/"$catkin_ws_name"/src
ln -s /home/"$user"/project/nmpc_racing /home/"$user"/"$catkin_ws_name"/src/.
cd /home/"$user"/"$catkin_ws_name"
catkin init
catkin build
