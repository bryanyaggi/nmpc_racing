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

'''
Get track data from CSV files
'''
def get_track_data(track_number='1'):
    track = {}
    directory = '/home/ubuntu/project/nmpc_racing/optimization/Map_track' + track_number + '/'
    csv_file = np.genfromtxt(directory + 'center_x_track' + track_number + '.csv', delimiter=',', dtype=float)
    track['center_x'] = csv_file[:].tolist()
    csv_file = np.genfromtxt(directory + 'center_y_track' + track_number + '.csv', delimiter=',', dtype=float)
    track['center_y'] = csv_file[:].tolist()
    csv_file = np.genfromtxt(directory + 'bound_x1_track' + track_number + '.csv', delimiter=',', dtype=float)
    track['bound_x1'] = csv_file[:].tolist()
    csv_file = np.genfromtxt(directory + 'bound_y1_track' + track_number + '.csv', delimiter=',', dtype=float)
    track['bound_y1'] = csv_file[:].tolist()
    csv_file = np.genfromtxt(directory + 'bound_x2_track' + track_number + '.csv', delimiter=',', dtype=float)
    track['bound_x2'] = csv_file[:].tolist()
    csv_file = np.genfromtxt(directory + 'bound_y2_track' + track_number + '.csv', delimiter=',', dtype=float)
    track['bound_y2'] = csv_file[:].tolist()

    return track

class Node:
    def __init__(self, track_number, controller='nmpc-baseline'):
        # Set up controller
        if controller == 'nmpc-baseline':
            self.controller = NMPC()
            callback = self.runNmpcBaseline
        elif controller == 'nmpc-kinematic':
            self.controller = NMPCKinematic()
            callback = self.runNmpcKinematic
        elif controller == 'mpc-kinematic':
            self.controller = MPCKinematic()
            callback = self.runMpcKinematic
        elif controller == 'mpc-dynamic':
            self.controller = MPCDynamic()
            callback = self.runMpcDynamic

        # Open log file
        f = open('/home/ubuntu/project/nmpc_racing/data/' + controller + '_' + track_number + '.csv', 'w')
        self.writer = csv.writer(f)

        # Command publishers
        LRW_topic = '/car_1/left_rear_wheel_velocity_controller/command'
        RRW_topic = '/car_1/right_rear_wheel_velocity_controller/command'
        LFW_topic = '/car_1/left_front_wheel_velocity_controller/command'
        RFW_topic = '/car_1/right_front_wheel_velocity_controller/command'
        LSH_topic = '/car_1/left_steering_hinge_position_controller/command'
        RSH_topic = '/car_1/right_steering_hinge_position_controller/command'
        self.pub_vel_LRW = rospy.Publisher(LRW_topic, Float64, queue_size = 1)
        self.pub_vel_RRW = rospy.Publisher(RRW_topic, Float64, queue_size = 1)
        self.pub_vel_LFW = rospy.Publisher(LFW_topic, Float64, queue_size = 1)
        self.pub_vel_RFW = rospy.Publisher(RFW_topic, Float64, queue_size = 1)
        self.pub_pos_LSH = rospy.Publisher(LSH_topic, Float64, queue_size = 1)
        self.pub_pos_RSH = rospy.Publisher(RSH_topic, Float64, queue_size = 1)

        # RViz publishers
        self.pub_target_point = rospy.Publisher('/car_1/target_point', PointStamped, queue_size=1)
        self.pub_rollout_path = rospy.Publisher('/car_1/rollout_path', Path, queue_size=1)
        self.pub_rollout_path_projection = rospy.Publisher('/car_1/rollout_path_projection', Path, queue_size=1, latch=True)
        self.pub_centerline = rospy.Publisher('/road/centerline', Path, queue_size=1, latch=True)
        self.pub_bound1 = rospy.Publisher('/road/bound1', Path, queue_size=1, latch=True)
        self.pub_bound2 = rospy.Publisher('/road/bound2', Path, queue_size=1, latch=True)
        self.pub_east = rospy.Publisher('/east', Path, queue_size=1, latch=True)
        self.pub_centerline = rospy.Publisher('/car_1/centerline', Path, queue_size=1)
        
        # State publisher
        self.pub_state = rospy.Publisher('/car_1/state', Float64MultiArray, queue_size=1)
        self.state_msg = Float64MultiArray()
        self.state_msg.data = [0.0] * 6

        self.steering_angle_msg = Float64()
        self.velocity_msg = Float64()

        self.target_point_display = PointStamped()
        self.target_point_display.header.frame_id = 'world'

        self.rate = rospy.Rate(30)
        self.tick = 0 # for delay
        
        # Get track data
        self.track = get_track_data(track_number)
        self.publish_track()
        
        # Enable controller
        rospy.Subscriber("/car_1/ground_truth", Odometry, callback, queue_size=1)
        
    def publish_track(self):
        self.pub_centerline.publish(xy_to_path(self.track['center_x'], self.track['center_y']))
        self.pub_bound1.publish(xy_to_path(self.track['bound_x1'], self.track['bound_y1']))
        self.pub_bound2.publish(xy_to_path(self.track['bound_x2'], self.track['bound_y2']))
        self.pub_east.publish(xy_to_path([i for i in range(5)], [0 for i in range(5)]))

    def control_vehicle(self, velocity, steering_angle):
        self.velocity_msg.data = velocity
        self.steering_angle_msg.data = steering_angle
        self.pub_pos_LSH.publish(self.steering_angle_msg)
        self.pub_pos_RSH.publish(self.steering_angle_msg)
        self.pub_vel_LFW.publish(self.velocity_msg)
        self.pub_vel_RFW.publish(self.velocity_msg)
        self.pub_vel_LRW.publish(self.velocity_msg)
        self.pub_vel_RRW.publish(self.velocity_msg)

    def delay(self, ticks=100):
        if self.tick < ticks:
            self.tick += 1
            self.rate.sleep()
            return True
        return False

    def runNmpcBaseline(self, data):
        if self.delay():
            return

        now_rostime = rospy.get_rostime()
        print('NMPC-Baseline')
        #rospy.loginfo("Current time %f", now_rostime.secs)

        # Get state
        self.controller.get_state(data)
        self.state_msg.data = [x for x in self.controller.x0]
        self.pub_state.publish(self.state_msg)

        # Get rollout path projected on centerline for optimization track constraint
        self.controller.project_rollout_to_centerline(self.track['center_x'], self.track['center_y'])
        self.pub_rollout_path_projection.publish(xy_to_path(self.controller.proj_center_X, self.controller.proj_center_Y))

        # Get target point
        self.controller.get_target_point(self.track['center_x'], self.track['center_y'])
        self.target_point_display.point.x = self.controller.target_point[0]
        self.target_point_display.point.y = self.controller.target_point[1]
        self.pub_target_point.publish(self.target_point_display)

        # Solve optimization
        self.controller.solve_optimization()
        
        # Simulate dynamics using optimization result u*
        self.controller.trajectory_rollout()

        # Control vehicle
        velocity = self.controller.u_cl1 * 100
        steering_angle = self.controller.u_cl2
        self.control_vehicle(velocity, steering_angle)

        # Print lateral acceleration
        #a_lat = self.controller.model.lateral_acceleration(self.controller.x0, (self.controller.u_cl1, self.controller.u_cl2))
        #print(a_lat)

        self.pub_rollout_path.publish(xy_to_path(self.controller.xx1, self.controller.xx2))

        # Update log
        row = [now_rostime, self.controller.x0[0], self.controller.x0[1], self.controller.x0[2], self.controller.x03,
                self.controller.x0[4], self.controller.x0[5], self.controller.u_cl1, self.controller.u_cl2,
                self.controller.elapsed]
        self.writer.writerow(row)
        
        self.rate.sleep()

    def runNmpcKinematic(self, data):
        now_rostime = rospy.get_rostime()
        rospy.loginfo("Current time %f", now_rostime.secs)

        # Get state
        self.controller.get_state(data)
        self.state_msg.data = [x for x in self.controller.x0]
        self.pub_state.publish(self.state_msg)

        # Get rollout path projected on centerline for optimization track constraint
        self.controller.project_rollout_to_centerline(self.track['center_x'], self.track['center_y'])
        self.pub_rollout_path_projection.publish(xy_to_path(self.controller.proj_center_X, self.controller.proj_center_Y))

        # Get target point
        self.controller.get_target_point(self.track['center_x'], self.track['center_y'])
        self.target_point_display.point.x = self.controller.target_point[0]
        self.target_point_display.point.y = self.controller.target_point[1]
        self.pub_target_point.publish(self.target_point_display)

        # Solve optimization
        self.controller.solve_optimization()
        
        # Simulate dynamics using optimization result u*
        self.controller.trajectory_rollout()

        # Control vehicle
        velocity = self.controller.u_cl1 * 20
        steering_angle = self.controller.u_cl2
        self.control_vehicle(velocity, steering_angle)

        self.pub_rollout_path.publish(xy_to_path(self.controller.xx1, self.controller.xx2))

        # Update log
        row = [now_rostime, self.controller.x0[0], self.controller.x0[1], self.controller.x0[2],
                self.controller.u_cl1, self.controller.u_cl2, self.controller.elapsed]
        self.writer.writerow(row)
        
        self.rate.sleep()

    '''
    TODO: Not working
    '''
    def runMpcDynamic(self, data):
        now_rostime = rospy.get_rostime()
        rospy.loginfo("Current time %f", now_rostime.secs)

        # Get state
        self.controller.get_state(data)
        self.state_msg.data = self.controller.state.tolist()
        self.pub_state.publish(self.state_msg)

        # Get operating points
        self.controller.get_operating_points(self.track['center_x'], self.track['center_y'])
        
        # Get rollout path projected on centerline for optimization track constraint
        self.controller.project_rollout_to_centerline(self.track['center_x'], self.track['center_y'])
        self.pub_rollout_path_projection.publish(xy_to_path(self.controller.proj_center[0],
            self.controller.proj_center[1]))

        # Get target point
        self.controller.get_target_point(self.track['center_x'], self.track['center_y'])
        self.controller.get_target_point(self.track['center_x'], self.track['center_y'])
        self.target_point_display.point.x = self.controller.target_point[0]
        self.target_point_display.point.y = self.controller.target_point[1]
        self.pub_target_point.publish(self.target_point_display)

        # Solve optimization
        self.controller.solve_optimization()

        # Trajectory rollout
        self.controller.trajectory_rollout()
        self.controller.iter += 1

        velocity = self.controller.rollout_controls[0, 0] * 100
        steering_angle = self.controller.rollout_controls[1, 0]
        self.control_vehicle(velocity, steering_angle)

        self.pub_rollout_path.publish(xy_to_path(self.controller.rollout_states[0],
            self.controller.rollout_states[1]))

        row = [now_rostime, self.controller.state[0], self.controller.state[1], self.controller.state[2],
                self.controller.x03, self.controller.state[4], self.controller.state[5],
                self.controller.rollout_controls[0, 0], self.controller.rollout_controls[0, 1]]
        self.writer.writerow(row)

        self.rate.sleep()

    def runMpcKinematic(self, data):
        if self.delay():
            return

        now_rostime = rospy.get_rostime()
        rospy.loginfo("Current time %f", now_rostime.secs)

        # Get state
        self.controller.get_state(data)

        # Get operating points
        self.controller.get_operating_points(self.track['center_x'], self.track['center_y'])
        
        # Get rollout path projected on centerline for optimization track constraint
        self.controller.project_rollout_to_centerline(self.track['center_x'], self.track['center_y'])
        self.pub_rollout_path_projection.publish(xy_to_path(self.controller.proj_center[0],
            self.controller.proj_center[1]))

        # Get target point
        self.controller.get_target_point(self.track['center_x'], self.track['center_y'])
        self.target_point_display.point.x = self.controller.target_point[0]
        self.target_point_display.point.y = self.controller.target_point[1]
        self.pub_target_point.publish(self.target_point_display)

        # Solve optimization
        self.controller.solve_optimization()

        # Rollout
        self.controller.trajectory_rollout()

        self.controller.iter += 1

        velocity = self.controller.rollout_controls[0, 0] * 20
        steering_angle = self.controller.rollout_controls[1, 0]
        self.control_vehicle(velocity, steering_angle)

        self.pub_rollout_path.publish(xy_to_path(self.controller.rollout_states[0], self.controller.rollout_states[1]))

        # Update log
        row = [now_rostime, self.controller.state[0], self.controller.state[1], self.controller.state[2],
                self.controller.rollout_controls[0, 0], self.controller.rollout_controls[0, 1]]
        self.writer.writerow(row)

        self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('controller_node', anonymous=True)
    node = Node(track_number='1', controller='nmpc-baseline')
    #node = Node(track_number='1', controller='nmpc-kinematic')
    #node = Node(track_number='1', controller='mpc-dynamic')
    #node = Node(track_number='1', controller='mpc-kinematic')
    rospy.spin()
