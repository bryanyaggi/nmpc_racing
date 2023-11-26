import casadi.casadi as cs
import cvxpy as cp
import math
import numpy as np
import time

from models import DynamicModel, KinematicModel
from track_utils import *

import sys
sys.path.insert(1, "/home/ubuntu/project/nmpc_racing/optimization/PANOC_DYNAMIC_MOTOR_MODEL/dynamic_my_optimizer/dynamic_racing_target_point")
import dynamic_racing_target_point

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
        self.proj_center = None
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

    def get_operating_points(self, center_x, center_y):
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
            self.proj_center = np.array((2, self.horizon))
            print(len(self.proj_center_x))
            self.proj_center[0] = self.proj_center_x[:-1]
            self.proj_center[1] = self.proj_center_y[:-1]
        else:
            proj_center = find_the_center_line(self.xx1[1:], self.xx2[1:], center_x, center_y)
            self.proj_center[0] = proj_center[0]
            self.proj_center[1] = proj_center[1]

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
        #constraints += [x[:2, i] <= ] # stay on track

        # control change cost
        if i == 0:
            cost += cp.quad_form(u[:, i] - prev_control, Q2)
        else:
            cost += cp.quad_form(u[:, i] - u[:, i - 1], Q2)

    cost += cp.quad_form(x[:2, horizon] - target_point, Q1) # final point cost

    constraints += [x[:, 0] == state] # initial state
    constraints += [u[0, :] >= 0] # velocity limits
    constraints += [u[0, :] <= 5]
    constraints += [u[1, :] >= -math.pi / 6] # steering angle limits
    constraints += [u[1, :] <= math.pi / 6]

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    return problem.status, u.value

    # return controls?
