#!/usr/bin/env python3
from __future__ import division

import sys
sys.path.insert(1, "/home/ubuntu/project/nmpc_racing/optimization/PANOC_DYNAMIC_MOTOR_MODEL/dynamic_my_optimizer/dynamic_racing_target_point")
import dynamic_racing_target_point

from models import DynamicModel, KinematicModel

import rospy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64, Float64MultiArray
import time
import casadi.casadi as cs
import numpy as np
import math
import csv
import cvxpy as cp

import unittest
import matplotlib.pyplot as plt

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
 
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
 
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
 
    return roll_x, pitch_y, yaw_z # in radians

'''
Find nearest point on centerline for each point in path
'''
def find_the_center_line(X_fut, Y_fut, center_x, center_y):
    dist_x = np.zeros(len(center_x))
    dist_y = np.zeros(len(center_x))
    N = len(X_fut)
    r = np.zeros((N, len(center_x)))
    center_x_proj = np.zeros(N)
    center_y_proj = np.zeros(N)

    for j in range(len(X_fut)):
        dist_x = (X_fut[j] - center_x)**2
        dist_y = (Y_fut[j] - center_y)**2
        r = dist_x+dist_y
        x = np.argmin(r)
        center_x_proj[j] = center_x[x]
        center_y_proj[j] = center_y[x]       	
        
    return center_x_proj, center_y_proj

'''
Find target point on centerline
'''
def perception_target_point(X_odom,Y_odom,center_x,center_y,a):
    center_x = np.concatenate((center_x, center_x))
    center_y = np.concatenate((center_y, center_y))
    dist_x = np.empty(len(center_x))
    dist_y = np.empty(len(center_x))
    r = np.empty(len(center_x))

    dist_x = (X_odom - center_x)**2
    dist_y = (Y_odom - center_y)**2
    r = dist_x+dist_y;

    x = np.argmin(r) # TODO: This is already done in previous function
    target_point_x = center_x[x+a]
    target_point_y = center_y[x+a]

    return target_point_x, target_point_y

def get_closest_point_on_centerline(x, y, center_x, center_y):
    dx = (x - np.array(center_x)) ** 2
    dy = (y - np.array(center_y)) ** 2
    dr = dx + dy

    index = np.argmin(dr)
    
    return index

'''
Returns equally spaced points along centerline
'''
def sample_centerline(start_x, start_y, center_x, center_y, points_in=91, points_out=50):
    # Get points along centerline
    start_i = get_closest_point_on_centerline(start_x, start_y, center_x, center_y)
    if start_i + points_in < len(center_x):
        segment_center_x = center_x[start_i:start_i+points_in]
        segment_center_y = center_y[start_i:start_i+points_in]
    else:
        end_i = points_in - (len(center_x) - 1 - start_i)
        segment_center_x = center_x[start_i:] + center_x[:end_i]
        segment_center_y = center_y[start_i:] + center_y[:end_i]

    segment_center_x = np.array(segment_center_x)
    segment_center_y = np.array(segment_center_y)
    dx, dy = segment_center_x[1:] - segment_center_x[:-1], segment_center_y[1:] - segment_center_y[:-1]
    ds = np.array((0, *np.sqrt(dx**2 + dy**2))) # distance along path between points
    s = np.cumsum(ds) # distance from start

    spacing = s[-1] / points_out
    x = np.interp(np.arange(0, s[-1] + spacing, spacing), s, segment_center_x)
    y = np.interp(np.arange(0, s[-1] + spacing, spacing), s, segment_center_y)

    return x, y

def get_path_yaw(path_x, path_y):
    dx, dy = path_x[1:] - path_x[:-1], path_y[1:] - path_y[:-1]
    yaw = np.arctan2(dy, dx)

    return yaw
    
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

class NMPC:
    def __init__(self):
        self.N = 50
        self.T = 0.033
        self.n_states = 6
        self.n_controls = 2
        self.mpciter = 0
        self.u_cl1 = 0
        self.u_cl2 = 0
        self.xx1 = np.empty(self.N + 1)
        self.xx2 = np.empty(self.N + 1)
        self.xx3 = np.empty(self.N + 1)
        self.xx4 = np.empty(self.N + 1)
        self.xx5 = np.empty(self.N + 1)
        self.xx6 = np.empty(self.N + 1)
        self.x0 = [0, 0, 0, 1, 0, 0] # initial conditions
        self.x03 = 1
        self.guess = [0.0] * (2 * self.N)
        self.theta2unwrap = []
        self.proj_center_X = None
        self.proj_center_Y = None
        self.target_point = None
        self.elapsed = None

        self.model = DynamicModel()
        self.solver = dynamic_racing_target_point.solver()
    
    def get_state(self, data):
        self.x0 = [0.0] * 6

        pose = data.pose.pose
        self.x0[0] = pose.position.x
        self.x0[1] = pose.position.y
        orientation_euler = euler_from_quaternion(pose.orientation.x, pose.orientation.y, pose.orientation.z,
                pose.orientation.w)
        theta = orientation_euler[2]
        self.theta2unwrap.append(theta)
        thetaunwrapped = np.unwrap(self.theta2unwrap)
        self.x0[2] = thetaunwrapped[-1]
        twist = data.twist.twist
        if self.mpciter < 15:
            self.x0[3] = 3
        else:
            self.x0[3] = twist.linear.x * math.cos(self.x0[2]) + twist.linear.y * math.sin(self.x0[2])
        self.x03 = twist.linear.x * math.cos(self.x0[2]) + twist.linear.y * math.sin(self.x0[2])
        self.x0[4] = twist.linear.y * math.cos(self.x0[2]) - twist.linear.x * math.sin(self.x0[2])
        self.x0[5] = twist.angular.z

    def project_rollout_to_centerline(self, center_x, center_y):
        if self.mpciter < 1:
            proj_center = find_the_center_line(np.linspace(0, 1, self.N), np.zeros(self.N), center_x, center_y)
            self.proj_center_X = proj_center[0]
            self.proj_center_Y = proj_center[1]
        else:
            #proj_center = find_the_center_line(self.xx1[1:self.N+1], self.xx2[1:self.N+1], center_x, center_y)
            proj_center = find_the_center_line(self.xx1[1:], self.xx2[1:], center_x, center_y)
            self.proj_center_X = proj_center[0]
            self.proj_center_Y = proj_center[1]

    def get_target_point(self, center_x, center_y):
        self.target_point = perception_target_point(self.x0[0], self.x0[1], center_x, center_y, 90)

    def ncvxopt(self):
        parameter = []
        for i in range(self.n_states):
            parameter.append(self.x0[i])
        # preU
        parameter.append(self.u_cl1)
        parameter.append(self.u_cl2)
        # target point
        parameter.append(self.target_point[0])
        parameter.append(self.target_point[1])
        # center line projection
        for i in range(self.N):
            parameter.append(self.proj_center_X[i])
            parameter.append(self.proj_center_Y[i])

        now = time.time()
        result = self.solver.run(p=[parameter[i] for i in range(self.n_states + self.n_controls + 2 + 2 * self.N)],
                initial_guess=[self.guess[i] for i in range (self.n_controls * self.N)])

        return result

    def solve_optimization(self):
        t0 = time.time()
        result = self.ncvxopt()
        self.elapsed = time.time() - t0
        print(self.elapsed)
        u_star = np.full(self.n_controls * self.N, result.solution)
        self.guess = u_star
        self.u_cl1 = u_star[0]
        self.u_cl2 = u_star[1]

    def trajectory_rollout(self):
        self.model.rollout(self.x0, self.guess, self.N, self.T, self.xx1, self.xx2, self.xx3, self.xx4, self.xx5, self.xx6)
        self.mpciter += 1

class MPC:
    def __init__(self):
        self.model = KinematicModel()
        self.horizon = 50
        self.dt = 0.033
        self.rollout_controls = np.zeros((2, self.horizon))
        self.rollout_states = np.zeros((3, self.horizon + 1))
        self.operating_point_states = None
        self.operating_point_controls = None
        self.proj_center_x = None
        self.proj_center_y = None
        self.target_point = None
        self.state = np.zeros(3)
        self.iter = 0

    def get_state(self, odom):
        self.state[0] = odom.pose.pose.position.x
        self.state[1] = odom.pose.pose.position.y
        _, _, yaw = euler_from_quaternion(odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z, odom.pose.pose.orientation.w)
        if yaw < -math.pi:
            yaw += 2 * math.pi
        elif yaw > math.pi:
            yaw -= 2 * math.pi
        self.state[2] = yaw

    def get_operating_points(self):
        if self.iter < 1:
            self.operating_point_states = np.zeros((3, self.horizon))
            path = sample_centerline(self.state[0], self.state[1], center_x, center_y)
            self.operating_point_states[0], self.operating_point_states[1] = path[0][:-1], path[1][:-1]
            self.operating_point_states[2, :-1] = get_path_yaw(self.operating_point_states[0],
                    self.operating_point_states[1])
            self.operating_point_states[2, -1] = self.operating_point_states[2, -2]
            self.operating_point_controls = np.ones((2, self.horizon))
            self.operating_point_controls[0] *= 3.0
            self.operating_point_controls[1] *= 0.0
        else:
            self.operating_point_states = self.rollout_states[:, 1:]
            self.operating_point_controls[:, :-1] = self.rollout_controls[:, 1:]
            self.operating_point_controls[:, -1] = self.rollout_controls[:, -1]

    def project_rollout_to_centerline(self, center_x, center_y):
        if self.iter < 1:
            '''
            proj_center = find_the_center_line(np.linspace(0, 1, self.N), np.zeros(self.N), center_x, center_y)
            self.proj_center_x = proj_center[0]
            self.proj_center_y = proj_center[1]
            '''
            self.proj_center_x, self.proj_center_y = sample_centerline(self.state[0], self.state[1], center_x, center_y)
        else:
            proj_center = find_the_center_line(self.xx1[1:], self.xx2[1:], center_x, center_y)
            self.proj_center_x = proj_center[0]
            self.proj_center_y = proj_center[1]

    def get_target_point(self, center_x, center_y):
        self.target_point = perception_target_point(self.state[0], self.state[1], center_x, center_y, 90)

    def solve_optimization(self):
        if self.iter < 1:
            prev_control = np.zeros(2)
        else:
            prev_control = self.rollout_controls[:, 0]
        t0 = time.time()
        status, self.rollout_controls = cvxopt(self.model, self.state, prev_control, self.target_point,
                self.operating_point_states, self.operating_point_controls)
        print('MPC optimization time: %f' %(time.time() - t0))
        print('MPC status: %s' %status)

    '''
    Updates rollout states using model and rollout controls
    '''
    def trajectory_rollout(self):
        self.model.rollout(self.state, self.rollout_controls, self.rollout_states)
    
    def run(self, odom):
        # Get state from odometry
        self.get_state(odom)

        # Get operating points
        self.get_operating_points()

        # Project rollout path to centerline
        self.project_rollout_to_centerline(center_x, center_y)

        # Get target point
        self.get_target_point(center_x, center_y)

        # Construct and solve optimization
        self.solve_optimization()

        # Trajectory rollout
        self.trajectory_rollout(self.state)

        self.iter += 1 # increment interation variable

        # Control vehicle

def cvxopt(model, state, prev_control, target_point, operating_point_states, operating_point_controls):
    horizon = operating_point_states.shape[1]
    x = cp.Variable((3, horizon + 1))
    u = cp.Variable((2, horizon))

    Q1 = np.eye(2) * 10
    Q2 = np.eye(2) * 10

    cost = 0
    constraints = []
    for i in range(horizon):
        yaw = operating_point_states[2, i]
        velocity = operating_point_controls[0, i]
        steering_angle = operating_point_controls[1, i]
        A, B, C = model.get_linear_model(yaw, velocity, steering_angle)
        constraints += [x[:, i + 1] == A @ x[:, i] + B @ u[:, i] + C] # dynamics

        # control change cost
        if i == 0:
            cost += cp.quad_form(u[:, i] - prev_control, Q2)
        else:
            cost += cp.quad_form(u[:, i] - u[:, i - 1], Q2)

    cost += cp.quad_form(x[:2, horizon - 1] - target_point, Q1) # final point cost

    constraints += [x[:, 0] == state] # initial state
    #constraints += # stay on track
    constraints += [u[0, :] >= 0] # velocity limits
    constraints += [u[0, :] <= 5]
    constraints += [u[1, :] >= -math.pi / 6] # steering angle limits
    constraints += [u[1, :] <= math.pi / 6]

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    return problem.status, u.value

    # return controls?

class Test(unittest.TestCase):
    def testSampleCenterline(self):
        x, y = 0, 0
        path_x, path_y = sample_centerline(x, y, center_x, center_y)
        print(path_x)
        print(path_y)

        fig, ax = plt.subplots()
        ax.plot(center_x, center_y, color='black')
        ax.plot(path_x, path_y, color='blue')
        plt.show()

    def testGetPathYaw(self):
        x, y = 0, 0
        path_x, path_y = sample_centerline(0, 0, center_x, center_y)
        print(path_x)
        print(path_y)

        yaw = get_path_yaw(path_x, path_y)
        print(yaw)

        fig, ax = plt.subplots()
        ax.plot(path_x, path_y)
        ax.axis('equal')
        plt.show()

# Get track data from CSV files
csv_file = np.genfromtxt('/home/ubuntu/project/nmpc_racing/optimization/Map_track3/center_x_track3.csv', 
                          delimiter=',', dtype=float)
center_x = csv_file[:].tolist()
csv_file = np.genfromtxt('/home/ubuntu/project/nmpc_racing/optimization/Map_track3/center_y_track3.csv', 
                          delimiter=',', dtype=float)
center_y = csv_file[:].tolist()
csv_file = np.genfromtxt('/home/ubuntu/project/nmpc_racing/optimization/Map_track3/bound_x1_track3.csv', 
                          delimiter=',', dtype=float)
bound_x1 = csv_file[:].tolist()
csv_file = np.genfromtxt('/home/ubuntu/project/nmpc_racing/optimization/Map_track3/bound_y1_track3.csv', 
                          delimiter=',', dtype=float)
bound_y1 = csv_file[:].tolist()
csv_file = np.genfromtxt('/home/ubuntu/project/nmpc_racing/optimization/Map_track3/bound_x2_track3.csv', 
                          delimiter=',', dtype=float)
bound_x2 = csv_file[:].tolist()
csv_file = np.genfromtxt('/home/ubuntu/project/nmpc_racing/optimization/Map_track3/bound_y2_track3.csv', 
                          delimiter=',', dtype=float)
bound_y2 = csv_file[:].tolist()

# Open log file
f = open('/home/ubuntu/project/nmpc_racing/data/race_DATA.csv', 'w')
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
mpc = MPC()

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
    mpc.get_operating_points()
    
    # Get rollout path projected on centerline for optimization track constraint
    nmpc.project_rollout_to_centerline(center_x, center_y)
    pub_rollout_path_projection.publish(xy_to_path(nmpc.proj_center_X, nmpc.proj_center_Y))
    #proj_x, proj_y = sample_centerline(state[0], state[1], center_x, center_y)
    #pub_center_path.publish(xy_to_path(proj_x, proj_y))

    # Get target point
    nmpc.get_target_point(center_x, center_y)
    mpc.get_target_point(center_x, center_y)
    target_point_display.point.x = nmpc.target_point[0]
    target_point_display.point.y = nmpc.target_point[1]
    pub_target_point.publish(target_point_display)

    # Solve optimization
    nmpc.solve_optimization()
    mpc.solve_optimization()
    print('Solved MPC optimization!')

    # Convert controls for MPC
    '''
    mpc.rollout_controls[0] = [5 * x for x in nmpc.guess[0::2]]
    mpc.rollout_controls[1] = nmpc.guess[1::2]
    '''
    mpc.trajectory_rollout()

    mpc.iter += 1

    # Simulate dynamics using optimization result u*
    nmpc.trajectory_rollout()

    velocity = nmpc.u_cl1 * 100
    steering_angle = nmpc.u_cl2
    control_vehicle(velocity, steering_angle)

    pub_rollout_path.publish(xy_to_path(nmpc.xx1, nmpc.xx2))
    pub_rollout_path_mpc.publish(xy_to_path(mpc.rollout_states[0], mpc.rollout_states[1]))

    # Update log
    row = [nmpc.x0[0], nmpc.x0[1], nmpc.x0[2], nmpc.x03, nmpc.x0[4], nmpc.x0[5], nmpc.elapsed, nmpc.u_cl1, nmpc.u_cl2, now_rostime]
    writer.writerow(row)

    rate.sleep()

if __name__ == '__main__':
    print("my node started")
    rospy.Subscriber("/car_1/ground_truth", Odometry, callback, queue_size=1)
    rospy.spin()
