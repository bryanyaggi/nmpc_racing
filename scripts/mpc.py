#!/usr/bin/env python3
from __future__ import division

import sys
sys.path.insert(1, "/home/ubuntu/project/nmpc_racing/optimization/PANOC_DYNAMIC_MOTOR_MODEL/dynamic_my_optimizer/dynamic_racing_target_point")
import dynamic_racing_target_point

from models import DynamicModel

import rospy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64, Float64MultiArray
import time
import casadi.casadi as cs
import numpy as np
import math
import csv

import unittest

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


############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
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
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
f = open('/home/ubuntu/project/nmpc_racing/data/race_DATA.csv', 'w')
writer = csv.writer(f)

rospy.init_node('my_mpc_node',anonymous = True)

LRW_topic   = '/car_1/left_rear_wheel_velocity_controller/command'
RRW_topic   = '/car_1/right_rear_wheel_velocity_controller/command'
LFW_topic   = '/car_1/left_front_wheel_velocity_controller/command'
RFW_topic   = '/car_1/right_front_wheel_velocity_controller/command'
LSH_topic   = '/car_1/left_steering_hinge_position_controller/command'
RSH_topic   = '/car_1/right_steering_hinge_position_controller/command'

# Command publishers
pub_vel_LRW = rospy.Publisher(LRW_topic, Float64, queue_size = 1)
pub_vel_RRW = rospy.Publisher(RRW_topic, Float64, queue_size = 1)
pub_vel_LFW = rospy.Publisher(LFW_topic, Float64, queue_size = 1)
pub_vel_RFW = rospy.Publisher(RFW_topic, Float64, queue_size = 1)
pub_pos_LSH = rospy.Publisher(LSH_topic, Float64, queue_size = 1)
pub_pos_RSH = rospy.Publisher(RSH_topic, Float64, queue_size = 1)

# RViz publishers
pub_target_point = rospy.Publisher('/car_1/target_point', PointStamped, queue_size=1)
pub_target_path = rospy.Publisher('/car_1/target_path', Path, queue_size=1)
pub_target_path_projection = rospy.Publisher('/car_1/target_path_projection', Path, queue_size=1, latch=True)
pub_centerline = rospy.Publisher('/road/centerline', Path, queue_size=1, latch=True)
pub_bound1 = rospy.Publisher('/road/bound1', Path, queue_size=1, latch=True)
pub_bound2 = rospy.Publisher('/road/bound2', Path, queue_size=1, latch=True)

steering_angle_msg = Float64()
velocity_msg = Float64()

target_point_display = PointStamped()
target_point_display.header.frame_id = 'world'

# Publish road
pub_centerline.publish(xy_to_path(center_x, center_y))
pub_bound1.publish(xy_to_path(bound_x1, bound_y1))
pub_bound2.publish(xy_to_path(bound_x2, bound_y2))

# State publisher
pub_state = rospy.Publisher('/car_1/state', Float64MultiArray, queue_size=1)
state_msg = Float64MultiArray()
state_msg.data = [0.0] * 6

rate = rospy.Rate(30)

#nmpc = NMPC()

'''
N = 50
T = 0.033

lr = 0.147;
lf = 0.178;
m  = 5.6922;
Iz  = 0.204;
df= 134.585
dr= 159.9198
cf= 0.085915
cr= 0.13364
bf= 9.2421
br= 17.7164
Cm1= 20
Cm2= 6.9281e-07
Cm3= 3.9901
Cm4= 0.66633
n_states = 6
n_controls = 2

mpciter = 0
u_cl1 = 0
u_cl2 = 0
xx1 = np.empty(N+1)
xx2 = np.empty(N+1)
xx3 = np.empty(N+1)
xx4 = np.empty(N+1)
xx5 = np.empty(N+1)
xx6 = np.empty(N+1)
x0 = [0, 0, 0, 1, 0, 0]		# initial conditions
guess = [0.0]*(2*N)
theta2unwrap = []
'''

#################################################################################################################################
#################################################################################################################################
#################################################################################################################################

def control_vehicle(velocity, steering_angle):
    velocity_msg.data = velocity
    steering_angle_msg.data = steering_angle
    pub_pos_LSH.publish(steering_angle_msg)
    pub_pos_RSH.publish(steering_angle_msg)
    pub_vel_LFW.publish(velocity_msg)
    pub_vel_RFW.publish(velocity_msg)
    pub_vel_LRW.publish(velocity_msg)
    pub_vel_RRW.publish(velocity_msg)

def cvxopt(model, state, prev_control, target_point, operating_points, horizon):
    x = cp.Variable((4, horizon + 1))
    u = cp.Variable((2, horizon))

    Q1 = np.eye(2) * 10
    Q2 = np.eye(2) * 10

    cost = 0
    constraints = []
    for i in range(H):
        A, B, C = model.get_linear_model()
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C] # dynamics

        # control change cost
        if i == 0:
            cost += cp.quad_form(u[:, t] - prev_control, Q2)
        else:
            cost += cp.quad_form(u[:, t] - u[:, t - 1], Q2)

    cost += cp.quad_form(x[:2, H - 1] - target_point, Q1) # final point cost

    constraints += [x[:, 0] == state] # initial state
    #constraints += # stay on track
    constraints += [u[0, :] >= 0] # velocity limits
    constraints += [u[0, :] <= 5]
    constraints += [u[0, :] >= -math.pi / 6] # steering angle limits
    constraints += [u[0, :] <= math.pi / 6]

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

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
            proj_center = find_the_center_line(self.xx1[1:self.N+1], self.xx2[1:self.N+1], center_x, center_y)
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

nmpc = NMPC()

def callback(data):
    now_rostime = rospy.get_rostime()
    rospy.loginfo("Current time %f", now_rostime.secs)

    # Get state
    nmpc.get_state(data)
    state_msg.data = [x for x in nmpc.x0]
    pub_state.publish(state_msg)
    
    # Get rollout path projected on centerline
    nmpc.project_rollout_to_centerline(center_x, center_y)
    
    pub_target_path_projection.publish(xy_to_path(nmpc.proj_center_X, nmpc.proj_center_Y))

    # Get target point
    nmpc.get_target_point(center_x, center_y)
    target_point_display.point.x = nmpc.target_point[0]
    target_point_display.point.y = nmpc.target_point[1]
    pub_target_point.publish(target_point_display)

    # Solve optimization
    result = nmpc.solve_optimization()

    # Simulate dynamics using optimization result u*
    nmpc.trajectory_rollout()

    velocity = nmpc.u_cl1 * 100
    steering_angle = nmpc.u_cl2
    control_vehicle(velocity, steering_angle)

    pub_target_path.publish(xy_to_path(nmpc.xx1, nmpc.xx2))

    # Update log
    row = [nmpc.x0[0], nmpc.x0[1], nmpc.x0[2], nmpc.x03, nmpc.x0[4], nmpc.x0[5], nmpc.elapsed, nmpc.u_cl1, nmpc.u_cl2, now_rostime]
    writer.writerow(row)

    rate.sleep()

def callback_(data):
    global mpciter
    global u_cl1, u_cl2, xx1, xx2, xx3, xx4, xx5, xx6, x0, guess

    now_rostime = rospy.get_rostime()
    rospy.loginfo("Current time %f", now_rostime.secs)

    # Get state
    x0 = get_state(data)
    x03 = x0[3]
    if mpciter < 15:
        x0[3] = 3
    state_msg.data = [x for x in x0]
    pub_state.publish(state_msg)
    
    # Get rollout path projected on centerline
    if mpciter < 1:
        proj_center = find_the_center_line(np.linspace(0, 1, N), np.zeros(N), center_x, center_y)
        proj_center_X = proj_center[0]
        proj_center_Y = proj_center[1]
        pub_target_path_projection.publish(xy_to_path(proj_center_X, proj_center_Y))
    else:
        proj_center = find_the_center_line(xx1[1:N+1], xx2[1:N+1], center_x, center_y)
        proj_center_X = proj_center[0]
        proj_center_Y = proj_center[1]
    
    #pub_target_path_projection.publish(xy_to_path(proj_center_X, proj_center_Y))

    # Get target point
    target_point = perception_target_point(x0[0], x0[1], center_x, center_y, 90)
    target_point_display.point.x = target_point[0]
    target_point_display.point.y = target_point[1]
    pub_target_point.publish(target_point_display)

    # Solve optimization
    t0 = time.time()
    result = ncvxopt(x0, proj_center_X, proj_center_Y, target_point, guess)
    elapsed = time.time() - t0
    print(elapsed)

    u_star = np.full(n_controls*N,result.solution)
    guess = u_star

    u_cl1 = u_star[0]
    u_cl2 = u_star[1]

    # Simulate dynamics using optimization result u*
    rollout()

    velocity = u_cl1 * 100
    steering_angle = u_cl2
    control_vehicle(velocity, steering_angle)

    pub_target_path.publish(xy_to_path(xx1, xx2))

    # Update log
    row = [x0[0], x0[1], x0[2], x03, x0[4], x0[5],elapsed,u_cl1,u_cl2,now_rostime]
    writer.writerow(row)

    mpciter = mpciter+1
    rate.sleep()

if __name__ == '__main__':
    print("my node started")
    rospy.Subscriber("/car_1/ground_truth", Odometry, callback, queue_size=1)
    rospy.spin()
