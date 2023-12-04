#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64, Float64MultiArray
import time
import numpy as np
import math
import csv

from controllers import *

import unittest
 
'''
Create path message from path x, y coordinates
'''
def xy_to_path(xs, ys, frame='world'):
    path = Path()
    path.header.frame_id = frame

    for i in range(len(xs)):
        ps = PoseStamped()
        ps.pose.position.x = xs[i]
        ps.pose.position.y = ys[i]
        path.poses.append(ps)

    return path

def control_vehicle(velocity, steering_angle):
    velocity_msg.data = velocity
    steering_angle_msg.data = steering_angle
    pub_pos_LSH.publish(steering_angle_msg)
    pub_pos_RSH.publish(steering_angle_msg)
    pub_vel_LFW.publish(velocity_msg)
    pub_vel_RFW.publish(velocity_msg)
    pub_vel_LRW.publish(velocity_msg)
    pub_vel_RRW.publish(velocity_msg)

'''
Get track data from CSV files
'''
def get_track_data(track_number='1'):
    directory = '/home/ubuntu/project/nmpc_racing/optimization/Map_track' + track_number + '/'
    csv_file = np.genfromtxt(directory + 'center_x_track' + track_number + '.csv', delimiter=',', dtype=float)
    center_x = csv_file[:].tolist()
    csv_file = np.genfromtxt(directory + 'center_y_track' + track_number + '.csv', delimiter=',', dtype=float)
    center_y = csv_file[:].tolist()
    csv_file = np.genfromtxt(directory + 'bound_x1_track' + track_number + '.csv', delimiter=',', dtype=float)
    bound_x1 = csv_file[:].tolist()
    csv_file = np.genfromtxt(directory + 'bound_y1_track' + track_number + '.csv', delimiter=',', dtype=float)
    bound_y1 = csv_file[:].tolist()
    csv_file = np.genfromtxt(directory + 'bound_x2_track' + track_number + '.csv', delimiter=',', dtype=float)
    bound_x2 = csv_file[:].tolist()
    csv_file = np.genfromtxt(directory + 'bound_y2_track' + track_number + '.csv', delimiter=',', dtype=float)
    bound_y2 = csv_file[:].tolist()

    return center_x, center_y, bound_x1, bound_y1, bound_x2, bound_y2

track_number = '1'

# Get track data
center_x, center_y, bound_x1, bound_y1, bound_x2, bound_y2 = get_track_data(track_number)

# Open log file
#f = open('/home/ubuntu/project/nmpc_racing/data/race_DATA.csv', 'w')
f = open('/home/ubuntu/project/nmpc_racing/data/nmpck_' + track_number + '.csv', 'w')
writer = csv.writer(f)

rospy.init_node('my_mpc_node',anonymous = True)

LRW_topic = '/car_1/left_rear_wheel_velocity_controller/command'
RRW_topic = '/car_1/right_rear_wheel_velocity_controller/command'
LFW_topic = '/car_1/left_front_wheel_velocity_controller/command'
RFW_topic = '/car_1/right_front_wheel_velocity_controller/command'
LSH_topic = '/car_1/left_steering_hinge_position_controller/command'
RSH_topic = '/car_1/right_steering_hinge_position_controller/command'

# Command publishers
pub_vel_LRW = rospy.Publisher(LRW_topic, Float64, queue_size = 1)
pub_vel_RRW = rospy.Publisher(RRW_topic, Float64, queue_size = 1)
pub_vel_LFW = rospy.Publisher(LFW_topic, Float64, queue_size = 1)
pub_vel_RFW = rospy.Publisher(RFW_topic, Float64, queue_size = 1)
pub_pos_LSH = rospy.Publisher(LSH_topic, Float64, queue_size = 1)
pub_pos_RSH = rospy.Publisher(RSH_topic, Float64, queue_size = 1)

# RViz publishers
pub_target_point = rospy.Publisher('/car_1/target_point', PointStamped, queue_size=1)
pub_rollout_path = rospy.Publisher('/car_1/rollout_path', Path, queue_size=1)
pub_rollout_path_mpc = rospy.Publisher('/car_1/rollout_path_mpc', Path, queue_size=1)
pub_rollout_path_projection = rospy.Publisher('/car_1/rollout_path_projection', Path, queue_size=1, latch=True)
pub_rollout_path_projection_mpc = rospy.Publisher('/car_1/rollout_path_projection_mpc', Path, queue_size=1, latch=True)
pub_centerline = rospy.Publisher('/road/centerline', Path, queue_size=1, latch=True)
pub_bound1 = rospy.Publisher('/road/bound1', Path, queue_size=1, latch=True)
pub_bound2 = rospy.Publisher('/road/bound2', Path, queue_size=1, latch=True)
pub_east = rospy.Publisher('/east', Path, queue_size=1, latch=True)
pub_center_path = rospy.Publisher('/car_1/center_path', Path, queue_size=1)

steering_angle_msg = Float64()
velocity_msg = Float64()

target_point_display = PointStamped()
target_point_display.header.frame_id = 'world'

# Publish road
pub_centerline.publish(xy_to_path(center_x, center_y))
pub_bound1.publish(xy_to_path(bound_x1, bound_y1))
pub_bound2.publish(xy_to_path(bound_x2, bound_y2))
pub_east.publish(xy_to_path([i for i in range(5)], [0 for i in range(5)]))

# State publisher
pub_state = rospy.Publisher('/car_1/state', Float64MultiArray, queue_size=1)
state_msg = Float64MultiArray()
state_msg.data = [0.0] * 6

rate = rospy.Rate(30)

nmpc = NMPC()
nmpck = NMPCKinematic()
mpc = MPC()

def runNMPC(data):
    now_rostime = rospy.get_rostime()
    rospy.loginfo("Current time %f", now_rostime.secs)

    # Get state
    nmpc.get_state(data)
    state_msg.data = [x for x in nmpc.x0]
    pub_state.publish(state_msg)

    # Get rollout path projected on centerline for optimization track constraint
    nmpc.project_rollout_to_centerline(center_x, center_y)
    pub_rollout_path_projection.publish(xy_to_path(nmpc.proj_center_X, nmpc.proj_center_Y))

    # Get target point
    nmpc.get_target_point(center_x, center_y)
    target_point_display.point.x = nmpc.target_point[0]
    target_point_display.point.y = nmpc.target_point[1]
    pub_target_point.publish(target_point_display)

    # Solve optimization
    nmpc.solve_optimization()
    
    # Simulate dynamics using optimization result u*
    nmpc.trajectory_rollout()

    # Control vehicle
    velocity = nmpc.u_cl1 * 100
    steering_angle = nmpc.u_cl2
    control_vehicle(velocity, steering_angle)

    # Print lateral acceleration
    #a_lat = nmpc.model.lateral_acceleration(nmpc.x0, (nmpc.u_cl1, nmpc.u_cl2))
    #print(a_lat)

    pub_rollout_path.publish(xy_to_path(nmpc.xx1, nmpc.xx2))

    # Update log
    #row = [nmpc.x0[0], nmpc.x0[1], nmpc.x0[2], nmpc.x03, nmpc.x0[4], nmpc.x0[5], nmpc.elapsed, nmpc.u_cl1, nmpc.u_cl2, now_rostime]
    row = [now_rostime, nmpc.x0[0], nmpc.x0[1], nmpc.x0[2], nmpc.x03, nmpc.x0[4], nmpc.x0[5], nmpc.u_cl1, nmpc.u_cl2,
            nmpc.elapsed]
    writer.writerow(row)
    
    rate.sleep()

def runNMPCKinematic(data):
    now_rostime = rospy.get_rostime()
    rospy.loginfo("Current time %f", now_rostime.secs)

    # Get state
    nmpck.get_state(data)
    state_msg.data = [x for x in nmpck.x0]
    pub_state.publish(state_msg)

    # Get rollout path projected on centerline for optimization track constraint
    nmpck.project_rollout_to_centerline(center_x, center_y)
    pub_rollout_path_projection.publish(xy_to_path(nmpck.proj_center_X, nmpck.proj_center_Y))

    # Get target point
    nmpck.get_target_point(center_x, center_y)
    target_point_display.point.x = nmpck.target_point[0]
    target_point_display.point.y = nmpck.target_point[1]
    pub_target_point.publish(target_point_display)

    # Solve optimization
    nmpck.solve_optimization()
    
    # Simulate dynamics using optimization result u*
    nmpck.trajectory_rollout()

    # Control vehicle
    #velocity = nmpc.u_cl1 * 100
    velocity = nmpck.u_cl1 * 20
    steering_angle = nmpck.u_cl2
    control_vehicle(velocity, steering_angle)

    pub_rollout_path.publish(xy_to_path(nmpck.xx1, nmpck.xx2))

    # Update log
    row = [now_rostime, nmpck.x0[0], nmpck.x0[1], nmpck.x0[2], nmpck.u_cl1, nmpck.u_cl2, nmpck.elapsed]
    writer.writerow(row)
    
    rate.sleep()

def callback(data):
    now_rostime = rospy.get_rostime()
    rospy.loginfo("Current time %f", now_rostime.secs)

    # Get state
    nmpc.get_state(data)
    mpc.get_state(data)
    print(mpc.state)
    state_msg.data = [x for x in nmpc.x0]
    pub_state.publish(state_msg)

    # Get operating points
    mpc.get_operating_points(center_x, center_y)
    
    # Get rollout path projected on centerline for optimization track constraint
    nmpc.project_rollout_to_centerline(center_x, center_y)
    mpc.project_rollout_to_centerline(center_x, center_y)
    pub_rollout_path_projection.publish(xy_to_path(nmpc.proj_center_X, nmpc.proj_center_Y))
    pub_rollout_path_projection_mpc.publish(xy_to_path(mpc.proj_center[0], mpc.proj_center[1]))
    #proj_x, proj_y = sample_centerline(state[0], state[1], center_x, center_y)
    #pub_center_path.publish(xy_to_path(proj_x, proj_y))

    # Get target point
    nmpc.get_target_point(center_x, center_y)
    mpc.get_target_point(center_x, center_y)
    #target_point_display.point.x = nmpc.target_point[0]
    #target_point_display.point.y = nmpc.target_point[1]
    target_point_display.point.x = mpc.target_point[0]
    target_point_display.point.y = mpc.target_point[1]
    pub_target_point.publish(target_point_display)

    # Solve optimization
    nmpc.solve_optimization()
    #mpc.solve_optimization()
    #print('Solved MPC optimization!')

    # Convert controls for MPC
    '''
    mpc.rollout_controls[0] = [5 * x for x in nmpc.guess[0::2]]
    mpc.rollout_controls[1] = nmpc.guess[1::2]
    '''
    if mpc.iter > 100:
        mpc.trajectory_rollout()

    mpc.iter += 1

    # Simulate dynamics using optimization result u*
    nmpc.trajectory_rollout()

    velocity = 0
    if mpc.iter > 100:
        #velocity = nmpc.u_cl1 * 100
        velocity = mpc.rollout_controls[0, 0] * 20
    #steering_angle = nmpc.u_cl2
    steering_angle = mpc.rollout_controls[1, 0]
    control_vehicle(velocity, steering_angle)

    pub_rollout_path.publish(xy_to_path(nmpc.xx1, nmpc.xx2))
    pub_rollout_path_mpc.publish(xy_to_path(mpc.rollout_states[0], mpc.rollout_states[1]))

    # Update log
    row = [nmpc.x0[0], nmpc.x0[1], nmpc.x0[2], nmpc.x03, nmpc.x0[4], nmpc.x0[5], nmpc.elapsed, nmpc.u_cl1, nmpc.u_cl2, now_rostime]
    writer.writerow(row)

    rate.sleep()

if __name__ == '__main__':
    #rospy.Subscriber("/car_1/ground_truth", Odometry, callback, queue_size=1)
    #rospy.Subscriber("/car_1/ground_truth", Odometry, runNMPC, queue_size=1)
    rospy.Subscriber("/car_1/ground_truth", Odometry, runNMPCKinematic, queue_size=1)
    rospy.spin()
