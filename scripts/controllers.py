import casadi.casadi as cs
import cvxpy as cp
import math
import numpy as np
import time

from models import DynamicModel, KinematicModel, DynamicModelSymPy
from track_utils import *

import sys
sys.path.insert(1, "/home/ubuntu/project/nmpc_racing/optimization/PANOC_DYNAMIC_MOTOR_MODEL/dynamic_my_optimizer/dynamic_racing_target_point")
import dynamic_racing_target_point

sys.path.insert(1, "/home/ubuntu/project/nmpc_racing/optimization/kinematic_model/optimizer/nmpc_kinematic_optimizer")
import nmpc_kinematic_optimizer

import unittest

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

    '''
    Returns horizon points
    '''
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

        t0 = time.time()
        result = self.solver.run(p=[parameter[i] for i in range(self.n_states + self.n_controls + 2 + 2 * self.N)],
                initial_guess=[self.guess[i] for i in range (self.n_controls * self.N)])
        self.elapsed = time.time() - t0

        return result

    def solve_optimization(self):
        #t0 = time.time()
        result = self.ncvxopt()
        #self.elapsed = time.time() - t0
        print(self.elapsed)
        u_star = np.full(self.n_controls * self.N, result.solution)
        self.guess = u_star
        self.u_cl1 = u_star[0]
        self.u_cl2 = u_star[1]

    def trajectory_rollout(self):
        self.model.rollout(self.x0, self.guess, self.N, self.T, self.xx1, self.xx2, self.xx3, self.xx4, self.xx5, self.xx6)
        self.mpciter += 1

class NMPCKinematic:
    def __init__(self):
        self.N = 50
        self.T = 0.033
        self.n_states = 3
        self.n_controls = 2
        self.mpciter = 0
        self.u_cl1 = 0
        self.u_cl2 = 0
        self.xx1 = np.empty(self.N + 1)
        self.xx2 = np.empty(self.N + 1)
        self.xx3 = np.empty(self.N + 1)
        self.x0 = [0, 0, 0] # initial conditions
        self.guess = [0.0] * (2 * self.N)
        self.rollout_controls = np.zeros((self.n_controls, self.N))
        self.rollout_states = np.zeros((self.n_states, self.N + 1))
        self.proj_center_X = None
        self.proj_center_Y = None
        self.target_point = None
        self.elapsed = None

        self.model = KinematicModel()
        self.solver = nmpc_kinematic_optimizer.solver()
    
    def get_state(self, odom):
        self.x0[0] = odom.pose.pose.position.x
        self.x0[1] = odom.pose.pose.position.y
        _, _, yaw = euler_from_quaternion(odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z, odom.pose.pose.orientation.w)
        if yaw < -math.pi:
            yaw += 2 * math.pi
        elif yaw > math.pi:
            yaw -= 2 * math.pi
        self.x0[2] = yaw

    '''
    Returns horizon points
    '''
    def project_rollout_to_centerline(self, center_x, center_y):
        if self.mpciter < 1:
            proj_center = find_the_center_line(np.linspace(0, 1, self.N), np.zeros(self.N), center_x, center_y)
            self.proj_center_X = proj_center[0]
            self.proj_center_Y = proj_center[1]
        else:
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
        # Update multidimensional array with one-dimensional guess
        self.rollout_controls = self.guess.reshape((self.n_controls, self.N), order='F')

        self.model.rollout(self.x0, self.rollout_controls, self.rollout_states)
        
        # Update one-dimensional states from multidimensional array
        self.xx1 = self.rollout_states[0]
        self.xx2 = self.rollout_states[1]
        self.xx3 = self.rollout_states[2]

        self.mpciter += 1

class MPCDynamic:
    def __init__(self):
        self.model = DynamicModelSymPy()
        self.horizon = 50
        self.lookahead = 90
        self.dt = 0.033
        self.n_states = 6
        self.n_controls = 2
        self.rollout_controls = np.zeros((self.n_controls, self.horizon))
        self.rollout_states = np.zeros((self.n_states, self.horizon + 1))
        self.operating_point_states = None
        self.operating_point_controls = None
        self.proj_center = None
        self.target_point = None
        self.state = np.zeros(self.n_states)
        self.x03 = None
        self.iter = 0

    def get_state(self, odom):
        if self.iter < 50:
            self.state[3] = 3
        self.state[0] = odom.pose.pose.position.x
        self.state[1] = odom.pose.pose.position.y
        _, _, yaw = euler_from_quaternion(odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z, odom.pose.pose.orientation.w)
        if yaw < -math.pi:
            yaw += 2 * math.pi
        elif yaw > math.pi:
            yaw -= 2 * math.pi
        self.state[2] = yaw
        if self.iter < 15:
            self.state[3] = 3
        else:
            self.state[3] = odom.twist.twist.linear.x * math.cos(yaw) + odom.twist.twist.linear.y * math.sin(yaw)
        self.x03 = odom.twist.twist.linear.x * math.cos(yaw) + odom.twist.twist.linear.y * math.sin(yaw) # actual value
        self.state[4] = odom.twist.twist.linear.y * math.cos(yaw) - odom.twist.twist.linear.x * math.sin(yaw)
        self.state[5] = odom.twist.twist.angular.z

    def get_operating_points(self, center_x, center_y):
        if self.iter < 1:
            self.operating_point_states = np.zeros((self.n_states, self.horizon))
            '''
            path = find_the_center_line(np.linspace(0, 1, self.horizon), np.zeros(self.horizon), center_x, center_y)
            self.operating_point_states[0], self.operating_point_states[1] = path[0], path[1]
            '''
            # Sample H+1 points, take first H
            path = sample_centerline(self.state[0], self.state[1], center_x, center_y, points_in=self.lookahead + 1,
                    points_out=self.horizon + 1)
            self.operating_point_states[0], self.operating_point_states[1] = path[0][:-1], path[1][:-1]
            # Calculate yaws
            self.operating_point_states[2, :-1] = get_path_yaw(self.operating_point_states[0],
                    self.operating_point_states[1])
            self.operating_point_states[2, -1] = self.operating_point_states[2, -2]
            # Velocities
            self.operating_point_states[3, :] = 1.0
            # Controls
            self.operating_point_controls = np.zeros((self.n_controls, self.horizon))
            self.operating_point_controls[0, :] = 0.5
        else:
            # Use rollout ignoring old first elements
            self.operating_point_states = self.rollout_states[:, 1:]
            #self.operating_point_controls[:, :-1] = self.rollout_controls[:, 1:]
            #self.operating_point_controls[:, -1] = self.rollout_controls[:, -1] # use last rollout control for last
            # operating point
            self.operating_point_states[4, :] = 0 # zero vy
            #self.operating_point_states[5, :] = 0 # zero omega
            self.operating_point_controls = self.rollout_controls # use previous solution
            self.operating_point_controls[1, :] = 0 # zero steering angle

    def project_rollout_to_centerline(self, center_x, center_y):
        #if self.iter < 1:
        if True:
            self.proj_center = np.zeros((2, self.horizon))
            '''
            path = find_the_center_line(np.linspace(0, 1, self.horizon), np.zeros(self.horizon), center_x, center_y)
            self.proj_center[0] = path[0]
            self.proj_center[1] = path[1]
            '''
            path = sample_centerline(self.state[0], self.state[1], center_x, center_y, points_in=self.lookahead + 1,
                    points_out=self.horizon + 1)
            self.proj_center[0] = path[0][1:]
            self.proj_center[1] = path[1][1:]
            
        else:
            path = find_the_center_line(self.rollout_states[0, 1:], self.rollout_states[1, 1:], center_x,
                    center_y)
            self.proj_center[0] = path[0]
            self.proj_center[1] = path[1]
    
    def get_target_point(self, center_x, center_y):
        self.target_point = perception_target_point(self.state[0], self.state[1], center_x, center_y, self.lookahead)
    
    def solve_optimization_(self):
        if self.iter < 1:
            prev_control = np.zeros(2)
        else:
            prev_control = self.rollout_controls[:, 0]

        t0 = time.time()
        status, solution = cvxopt_dynamic(self.model, self.state, prev_control, self.target_point, self.proj_center,
                self.operating_point_states, self.operating_point_controls)
        if status == 'optimal':
            self.rollout_controls = solution
        print('MPC optimization time: %f' %(time.time() - t0))
        print('MPC status: %s' %status)
    
    def solve_optimization(self):
        if self.iter < 1:
            self.define_parameters()
            self.construct_problem()
            prev_control = np.zeros(2)
        else:
            prev_control = self.rollout_controls[:, 0]

        # Update parameters
        if self.iter < 100:
            for i in range(self.horizon):
                self.A.value[:, i:i+self.n_states], self.B.value[:, i:i+self.n_controls], self.C.value[:, i] = \
                        self.model.get_linear_model(self.operating_point_states[:, 0], self.operating_point_controls[:, 0])
        else:
            for i in range(self.horizon):
                self.A.value[:, i:i+self.n_states], self.B.value[:, i:i+self.n_controls], self.C.value[:, i] = \
                        self.model.get_linear_model(self.operating_point_states[:, i], self.operating_point_controls[:, i])
        self.x.value = self.state
        self.u_prev.value = prev_control
        self.p_t.value = np.array(self.target_point)
        self.p_c.value = self.proj_center

        t0 = time.time()
        solver = cp.CLARABEL
        #solver = cp.SCIPY
        self.problem.solve(solver=solver, verbose=False, warm_start=True) # time_limit=1.0
        if self.problem.status == 'optimal':
            self.rollout_controls = self.U.value
        
        #print('MPC optimization time: %f' %(time.time() - t0))
        print('MPC status: %s' %self.problem.status)

    def define_parameters(self):
        shape = (self.n_states, self.n_states * self.horizon)
        self.A = cp.Parameter(shape, value=np.zeros(shape, dtype=np.float64))
        shape = (self.n_states, self.n_controls * self.horizon)
        self.B = cp.Parameter(shape, value=np.zeros(shape, dtype=np.float64))
        shape = (self.n_states, self.horizon)
        self.C = cp.Parameter(shape, value=np.zeros(shape, dtype=np.float64))
        self.x = cp.Parameter(self.n_states, value=np.zeros(self.n_states, dtype=np.float64)) # state
        self.u_prev = cp.Parameter(self.n_controls, value=np.zeros(self.n_controls, dtype=np.float64)) # previous control
        self.p_t = cp.Parameter(2, value=np.zeros(2, dtype=np.float64)) # target point
        shape = (2, self.horizon)
        self.p_c = cp.Parameter(shape, value=np.zeros(shape, dtype=np.float64)) # centerline projection

    def construct_problem(self):
        self.X = cp.Variable(self.rollout_states.shape,
                value=np.zeros(self.rollout_states.shape, dtype=np.float64))
        self.U = cp.Variable(self.rollout_controls.shape,
                value=np.zeros(self.rollout_controls.shape, dtype=np.float64))

        Q1 = np.eye(2) * 10
        Q2 = np.eye(2) * 10
        trackd = np.ones(2) * 4
        dd = 0.5
        dsteer = 0.025

        u_prev = self.u_prev

        cost = 0
        constraints = []
        for i in range(self.horizon):
            constraints += [self.X[:, i + 1] == \
                    self.A[:, i:i+self.n_states] @ self.X[:, i] \
                    + self.B[:, i:i+self.n_controls] @ self.U[:, i] \
                    + self.C[:, i]] # dynamics

            # control change cost
            cost += cp.quad_form(self.U[:, i] - u_prev, Q2)
            
            # control change constraints
            constraints += [self.U[0, i] - u_prev[0] >= -dd]
            constraints += [self.U[0, i] - u_prev[0] <= dd]
            constraints += [self.U[1, i] - u_prev[1] >= -dsteer]
            constraints += [self.U[1, i] - u_prev[1] <= dsteer]
            u_prev = self.U[:, i] # update previous control
            
            # stay on track
            constraints += [self.X[:2, i] <= self.p_c[:, i] + trackd]
            constraints += [self.X[:2, i] >= self.p_c[:, i] - trackd]

        cost += cp.quad_form(self.X[:2, self.horizon] - self.p_t, Q1) # final point cost

        constraints += [self.X[:, 0] == self.state] # initial state
        constraints += [self.X[3, :] >= 0] # vx limits for to prevent numerical issues
        constraints += [self.X[3, :] <= 5] 
        constraints += [self.U[0, :] >= 0] # d limits
        constraints += [self.U[0, :] <= 1]
        constraints += [self.U[1, :] >= -math.pi / 6] # steering angle limits
        constraints += [self.U[1, :] <= math.pi / 6]

        self.problem = cp.Problem(cp.Minimize(cost), constraints)
    
    '''
    Updates rollout states using model and rollout controls
    '''
    def trajectory_rollout(self):
        self.model.rollout(self.state, self.rollout_controls, self.rollout_states)

def cvxopt_dynamic(model, state, prev_control, target_point, proj_center, operating_point_states, operating_point_controls):
    horizon = operating_point_states.shape[1]
    x = cp.Variable((6, horizon + 1))
    u = cp.Variable((2, horizon))

    Q1 = np.eye(2) * 10
    Q2 = np.eye(2) * 10
    trackd = np.ones(2) * 2

    cost = 0
    constraints = []
    for i in range(horizon):
        A, B, C = model.get_linear_model(operating_point_states[:, i], operating_point_controls[:, i])
        constraints += [x[:, i + 1] == A @ x[:, i] + B @ u[:, i] + C] # dynamics

        if i == 0:
            cost += cp.quad_form(u[:, i] - prev_control, Q2) # control change cost
        else:
            cost += cp.quad_form(u[:, i] - u[:, i - 1], Q2) # control change cost
            # stay on track
            constraints += [x[:2, i] <= proj_center[:, i - 1] + trackd]
            constraints += [x[:2, i] >= proj_center[:, i - 1] - trackd]

    cost += cp.quad_form(x[:2, horizon] - target_point, Q1) # final point cost

    constraints += [x[:, 0] == state] # initial state
    constraints += [x[3, :] >= 1] # vx limit for to prevent numerical issues
    constraints += [u[0, :] >= 0] # d limits
    constraints += [u[0, :] <= 1]
    constraints += [u[1, :] >= -math.pi / 6] # steering angle limits
    constraints += [u[1, :] <= math.pi / 6]

    problem = cp.Problem(cp.Minimize(cost), constraints)
    solver = cp.CLARABEL
    #solver = cp.SCIPY
    problem.solve(solver=solver, verbose=True, time_limit=1.0) # TODO: Try warm start and intial guess

    return problem.status, u.value

class MPC:
    def __init__(self):
        self.model = KinematicModel()
        self.horizon = 50
        self.lookahead = 80
        self.dt = 0.033
        self.rollout_controls = np.zeros((2, self.horizon))
        self.rollout_states = np.zeros((3, self.horizon + 1))
        self.operating_point_states = None
        self.operating_point_controls = None
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

    '''
    Returns horizon points corresponding to steps k = 0 to H for linearizing dynamics
    '''
    def get_operating_points(self, center_x, center_y):
        if self.iter < 1:
            self.operating_point_states = np.zeros((3, self.horizon))

            # Sample H+1 points, take first H
            path = sample_centerline(self.state[0], self.state[1], center_x, center_y, points_in=self.lookahead + 1,
                    points_out=self.horizon + 1)
            self.operating_point_states[0], self.operating_point_states[1] = path[0][:-1], path[1][:-1]
            # Calculate yaws
            self.operating_point_states[2, :-1] = get_path_yaw(self.operating_point_states[0],
                    self.operating_point_states[1])
            self.operating_point_states[2, -1] = self.operating_point_states[2, -2]
            # Assign controls
            self.operating_point_controls = np.ones((2, self.horizon))
            self.operating_point_controls[0] *= 3.0
            self.operating_point_controls[1] *= 0.0
        else:
            # Use rollout ignoring old first elements
            self.operating_point_states = self.rollout_states[:, 1:]
            self.operating_point_controls[:, :-1] = self.rollout_controls[:, 1:]
            self.operating_point_controls[:, -1] = self.rollout_controls[:, -1] # use last rollout control for last
            # operating point

    '''
    Returns horizon points corresponding to steps k = 1 to H+1 for applying track constraint
    '''
    def project_rollout_to_centerline(self, center_x, center_y):
        #if self.iter < 1:
        if True:
            self.proj_center = np.zeros((2, self.horizon))
            '''
            path = find_the_center_line(np.linspace(0, 1, self.horizon), np.zeros(self.horizon), center_x, center_y)
            self.proj_center[0] = path[0]
            self.proj_center[1] = path[1]
            '''
            path = sample_centerline(self.state[0], self.state[1], center_x, center_y, points_in=self.lookahead + 1,
                    points_out=self.horizon + 1)
            self.proj_center[0] = path[0][1:]
            self.proj_center[1] = path[1][1:]
            
        else:
            path = find_the_center_line(self.rollout_states[0, 1:], self.rollout_states[1, 1:], center_x,
                    center_y)
            self.proj_center[0] = path[0]
            self.proj_center[1] = path[1]

    def get_target_point(self, center_x, center_y):
        self.target_point = perception_target_point(self.state[0], self.state[1], center_x, center_y, self.lookahead)

    def solve_optimization(self):
        if self.iter < 1:
            prev_control = np.zeros(2)
        else:
            prev_control = self.rollout_controls[:, 0]
        t0 = time.time()
        status, solution = cvxopt(self.model, self.state, prev_control, self.target_point, self.proj_center,
                self.operating_point_states, self.operating_point_controls)
        if status == 'optimal':
            self.rollout_controls = solution
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

def cvxopt(model, state, prev_control, target_point, proj_center, operating_point_states, operating_point_controls):
    horizon = operating_point_states.shape[1]
    x = cp.Variable((3, horizon + 1))
    u = cp.Variable((2, horizon))

    Q1 = np.eye(2) * 10
    Q2 = np.eye(2) * 10 # 10
    d = np.ones(2) * 2

    cost = 0
    constraints = []
    for i in range(horizon):
        yaw = operating_point_states[2, i]
        velocity = operating_point_controls[0, i]
        steering_angle = operating_point_controls[1, i]
        A, B, C = model.get_linear_model(yaw, velocity, steering_angle)
        constraints += [x[:, i + 1] == A @ x[:, i] + B @ u[:, i] + C] # dynamics

        if i == 0:
            cost += cp.quad_form(u[:, i] - prev_control, Q2) # control change cost
        else:
            cost += cp.quad_form(u[:, i] - u[:, i - 1], Q2) # control change cost
            # stay on track
            constraints += [x[:2, i] <= proj_center[:, i - 1] + d]
            constraints += [x[:2, i] >= proj_center[:, i - 1] - d]

    cost += cp.quad_form(x[:2, horizon] - target_point, Q1) # final point cost

    constraints += [x[:, 0] == state] # initial state
    constraints += [u[0, :] >= 0] # velocity limits
    constraints += [u[0, :] <= 5]
    constraints += [u[1, :] >= -math.pi / 6] # steering angle limits
    constraints += [u[1, :] <= math.pi / 6]

    problem = cp.Problem(cp.Minimize(cost), constraints)
    solver = cp.CLARABEL
    #solver = cp.SCIPY
    problem.solve(solver=solver, verbose=False) # TODO: Try warm start and intial guess

    return problem.status, u.value

    # return controls?

class TestOptimization(unittest.TestCase):
    def setUp(self):
        import csv

        csv_file = np.genfromtxt('/home/ubuntu/project/nmpc_racing/optimization/Map_track1/center_x_track1.csv', 
                          delimiter=',', dtype=float)
        self.center_x = csv_file[:].tolist()
        csv_file = np.genfromtxt('/home/ubuntu/project/nmpc_racing/optimization/Map_track1/center_y_track1.csv', 
                          delimiter=',', dtype=float)
        self.center_y = csv_file[:].tolist()
    
    def plot(self, start_point, target_point, rollout_points, projection_points):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        cx, cy = sample_centerline(start_point[0], start_point[1], self.center_x, self.center_y)
        ax.plot(cx, cy, color='yellow')
        
        ax.plot(projection_points[0], projection_points[1], color='orange')
        ax.scatter(start_point[0], start_point[1], color='green')
        ax.scatter(target_point[0], target_point[1], color='red')
        ax.plot(rollout_points[0], rollout_points[1], color='blue')

        ax.axis('equal')
        plt.show()

    def testNonconvex(self):
        nmpc = NMPC()
        nmpc.x0 = [0.0] * 6
        nmpc.x0[3] = 1
        for i in range(100):
            print(nmpc.x0)
            nmpc.project_rollout_to_centerline(self.center_x, self.center_y)
            nmpc.get_target_point(self.center_x, self.center_y)
            nmpc.solve_optimization()
            nmpc.trajectory_rollout()

            start_point = np.array([nmpc.x0[0], nmpc.x0[1]])
            target_point = np.array([nmpc.target_point[0], nmpc.target_point[1]])
            rollout_points = np.zeros((2, len(nmpc.xx1)))
            rollout_points[0] = nmpc.xx1
            rollout_points[1] = nmpc.xx2
            projection_points = np.zeros((2, len(nmpc.proj_center_X)))
            projection_points[0] = nmpc.proj_center_X
            projection_points[1] = nmpc.proj_center_Y
            self.plot(start_point, target_point, rollout_points, projection_points)
            
            nmpc.x0 = [nmpc.xx1[1], nmpc.xx2[1], nmpc.xx3[1], nmpc.xx4[1], nmpc.xx5[1], nmpc.xx6[1]]
        #self.plot(start_point, target_point, rollout_points, projection_points)

    def testConvex(self):
        mpc = MPC()
        mpc.state = np.zeros(3)
        for i in range(50):
            print(mpc.state)
            mpc.get_operating_points(self.center_x, self.center_y)
            mpc.project_rollout_to_centerline(self.center_x, self.center_y)
            mpc.get_target_point(self.center_x, self.center_y)
            mpc.solve_optimization()
            mpc.trajectory_rollout()
            mpc.iter += 1

            start_point = mpc.state[:2]
            target_point = mpc.target_point
            rollout_points = mpc.rollout_states[:2]
            projection_points = mpc.proj_center
            self.plot(start_point, target_point, rollout_points, projection_points)

            mpc.state = mpc.rollout_states[:, 1]

    def testConvexDynamic(self):
        mpcd = MPCDynamic()
        mpcd.state = np.zeros(6)
        mpcd.state[3] = 1
        for i in range(50):
            print(mpcd.state)
            mpcd.get_operating_points(self.center_x, self.center_y)
            mpcd.project_rollout_to_centerline(self.center_x, self.center_y)
            mpcd.get_target_point(self.center_x, self.center_y)
            mpcd.solve_optimization()
            mpcd.trajectory_rollout()
            mpcd.iter += 1

            start_point = mpcd.state[:2]
            target_point = mpcd.target_point
            rollout_points = mpcd.rollout_states[:2]
            projection_points = mpcd.proj_center
            self.plot(start_point, target_point, rollout_points, projection_points)

            mpcd.state = mpcd.rollout_states[:, 1]
