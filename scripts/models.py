import numpy as np
import math
import sympy as sp
from sympy.abc import phi, omega, delta
x, y, vx, vy, d, t = sp.symbols('x y vx vy d t') # remaining state and control variables

import unittest

class KinematicModel:
    def __init__(self):
        self.lr = 0.147
        self.lf = 0.178
        self.L = self.lr + self.lf

    def slip_angle(self, steering_angle):
        return math.atan(self.lr / (self.lr + self.lf) * math.tan(steering_angle))
    
    def lateral_acceleration(self, control):
        slip_angle = self.slip_angle(control[1])

        return control[0] ** 2 * math.sin(slip_angle) / self.lr

    def step(self, state, control, dt=0.033):
        slip_angle = math.atan(self.lr / (self.lr + self.lf) * math.tan(control[1]))

        next_state = np.zeros(state.shape)
        next_state[0] = state[0] + control[0] * math.cos(state[2] + slip_angle) * dt
        next_state[1] = state[1] + control[0] * math.sin(state[2] + slip_angle) * dt
        #next_state[2] = state[2] + control[0] * math.sin(slip_angle) / self.lr * dt
        next_state[2] = state[2] + control[0] * math.cos(slip_angle) * math.tan(control[1]) \
                / (self.lr + self.lf) * dt
        
        return next_state

    def get_linear_model_0(self, yaw, velocity, steering_angle=0, dt=0.033):
        slip_angle = math.atan(self.lr / (self.L) * math.tan(steering_angle))
        
        A = np.eye(3)
        A[0, 2] = -velocity * math.sin(yaw) * dt
        A[1, 2] = velocity * math.cos(yaw) * dt

        B = np.zeros((3, 2))
        B[0, 0] = math.cos(yaw) * dt
        B[1, 0] = math.sin(yaw) * dt

        C = np.zeros(3)
        C[0] = velocity * math.sin(yaw) * (yaw - slip_angle) * dt
        C[1] = velocity * math.cos(yaw) * (slip_angle - yaw) * dt
        C[2] = velocity * slip_angle / self.lr * dt

        return A, B, C
    
    def get_linear_model(self, yaw, velocity, steering_angle=0, dt=0.033):
        slip_angle = math.atan(self.lr / (self.L) * math.tan(steering_angle))
        #slip_angle = 0 # small angle approximation
        
        A = np.eye(3)
        A[0, 2] = -velocity * math.sin(yaw) * dt
        A[1, 2] = velocity * math.cos(yaw) * dt

        B = np.zeros((3, 2))
        B[0, 0] = math.cos(yaw) * dt
        B[1, 0] = math.sin(yaw) * dt
        B[2, 0] = math.tan(steering_angle) / self.L * dt
        B[2, 1] = velocity / (self.L * math.cos(steering_angle) ** 2) * dt

        C = np.zeros(3)
        C[0] = velocity * math.sin(yaw) * (yaw - slip_angle) * dt
        C[1] = velocity * math.cos(yaw) * (slip_angle - yaw) * dt
        C[2] = - velocity * steering_angle / (self.L * math.cos(steering_angle) ** 2) * dt

        return A, B, C

    def rollout(self, state, rollout_controls, rollout_states, dt=0.033):
        rollout_states[:, 0] = state

        for i in range(rollout_states.shape[1] - 1):
            rollout_states[:, i + 1] = self.step(rollout_states[:, i], rollout_controls[:, i], dt=dt)

class DynamicModel:
    def __init__(self):
        self.lr = 0.147
        self.lf = 0.178
        self.m = 5.6922
        self.Iz = 0.204
        self.df = 134.585
        self.dr = 159.9198
        self.cf = 0.085915
        self.cr = 0.13364
        self.bf = 9.2421
        self.br = 17.7164
        self.Cm1 = 20
        self.Cm2 = 6.9281e-07
        self.Cm3 = 3.9901
        self.Cm4 = 0.66633

    def step(self):
        pass

    def rollout(self, x0, guess, N, T, xx1, xx2, xx3, xx4, xx5, xx6):
        xx1[0] = x0[0]
        xx2[0] = x0[1]
        xx3[0] = x0[2]
        xx4[0] = x0[3]
        xx5[0] = x0[4]
        xx6[0] = x0[5]

        for i in range(N):
            xx1[i+1] = xx1[i] + T * (xx4[i] * math.cos(xx3[i]) - xx5[i] * math.sin(xx3[i]))
            xx2[i+1] = xx2[i] + T * (xx4[i] * math.sin(xx3[i]) + xx5[i] * math.cos(xx3[i]))
            xx3[i+1] = xx3[i] + T * (xx6[i])
            xx4[i+1] = xx4[i] + T * ((1/self.m) * ((self.Cm1 - self.Cm2 * xx4[i]) * guess[2*i] - self.Cm4 * xx4[i]**2 - self.Cm3 \
                        + ((self.Cm1 - self.Cm2 * xx4[i]) * guess[2*i] - self.Cm4 * xx4[i]**2 - self.Cm3) * math.cos(guess[2*i+1]) \
                        + self.m * xx5[i] * xx6[i] - self.df * math.sin(self.cf * math.atan(self.bf \
                        * (-math.atan((xx5[i] + self.lf * xx6[i]) / xx4[i]) + guess[2*i+1]))) * math.sin(guess[2*i+1])))
            xx5[i+1] = xx5[i] + T * ((1/self.m) * (((self.Cm1 - self.Cm2 * xx4[i]) * guess[2*i] - self.Cm4 * xx4[i]**2 - self.Cm3) \
                        * math.sin(guess[2*i+1]) - self.m * xx4[i] * xx6[i] + (self.df * math.sin(self.cf * math.atan(self.bf \
                        * (-math.atan((xx5[i] + self.lf * xx6[i]) / xx4[i]) + guess[2*i+1]))) * math.cos(guess[2*i+1]) + self.dr \
                        * math.sin(self.cr * math.atan(self.br * (-math.atan((xx5[i] - self.lr * xx6[i]) / xx4[i])))))))
            xx6[i+1] = xx6[i] + T * ((1/self.Iz) * (self.lf * ((self.Cm1 - self.Cm2 * xx4[i]) * guess[2*i] - self.Cm4 * xx4[i]**2 \
                        - self.Cm3) * math.cos(guess[2*i+1]) + self.lf * self.df * math.sin(self.cf * math.atan(self.bf \
                        * (-math.atan((xx5[i] + self.lf * xx6[i]) / xx4[i]) + guess[2*i+1]))) * math.cos(guess[2*i+1]) - self.lr * self.dr \
                        * math.sin(self.cr * math.atan(self.br * (-math.atan((xx5[i] - self.lr * xx6[i]) / xx4[i]))))))

        return xx1, xx2, xx3, xx4, xx5, xx6

    def lateral_acceleration(self, x0, control):
        a_lat = ((1/self.m) * (((self.Cm1 - self.Cm2 * x0[3]) * control[0] - self.Cm4 * x0[3]**2 - self.Cm3) \
                * math.sin(control[1]) - self.m * x0[3] * x0[5] + (self.df * math.sin(self.cf * math.atan(self.bf \
                * (-math.atan((x0[4] + self.lf * x0[5]) / x0[3]) + control[1]))) * math.cos(control[1]) + self.dr \
                * math.sin(self.cr * math.atan(self.br * (-math.atan((x0[4] - self.lr * x0[5]) / x0[3])))))))

        return a_lat

class DynamicModelSymPy:
    def __init__(self):
        self.create()

    def create(self):
        # Constants
        lr = 0.147
        lf = 0.178
        m = 5.6922
        Jz = 0.204
        Df = 134.585
        Dr = 159.9198
        Cf = 0.085915
        Cr = 0.13364
        Bf = 9.2421
        Br = 17.7164
        Cm1 = 20
        Cm2 = 6.9281e-07
        Cm3 = 3.9901
        Cm4 = 0.66633
        
        # Drivetrain model
        Fx = (Cm1 - Cm2 * vx) * d - Cm3 - Cm4 * vx ** 2

        # Pacejka model
        alphaf = -sp.atan((omega * lf + vy) / vx) + delta
        alphar = sp.atan((omega * lr - vy) / vx)

        Ffy = Df * sp.sin(Cf * sp.atan(Bf * alphaf))
        Fry = Dr * sp.sin(Cr * sp.atan(Br * alphar))

        f = sp.Matrix([vx * sp.cos(phi) - vx * sp.sin(phi),
                       vx * sp.sin(phi) + vy * sp.cos(phi),
                       omega,
                       (Fx - Ffy * sp.sin(delta) + Fx * sp.cos(delta) + m * vy * omega) / m,
                       (Fry + Ffy * sp.cos(delta) + Fx * sp.sin(delta) - m * vx * omega) / m,
                       (lf * Ffy * sp.cos(delta) + lf * Fx * sp.sin(delta) - lr * Fry) / Jz])

        # Lambda function for dynamics
        self.f = sp.lambdify([x, y, phi, vx, vy, omega, d, delta], f, 'numpy')

        # Jacobians
        Jx = f.jacobian([x, y, phi, vx, vy, omega])
        Ju = f.jacobian([d, delta])

        # Lambda functions for linearized matrices
        args = [x, y, phi, vx, vy, omega, d, delta, t]
        self.A = sp.lambdify(args, sp.eye(6) + Jx * t, 'numpy')
        self.B = sp.lambdify(args, Ju * t, 'numpy')
        self.C = sp.lambdify(args,
                (f - Jx * sp.Matrix([x, y, phi, vx, vy, omega]) - Ju * sp.Matrix([d, delta])) * t,
                'numpy')
        
    '''
    state is operating point state
    control is operating point control
    '''
    def get_linear_model(self, state, control, dt=0.033):
        '''    
        A = self.A.subs({x:state[0], y:state[1], phi:state[2], vx:state[3], vy:state[4], omega:state[5],
                         d:control[0], delta:control[1], t:0.033})
        B = self.B.subs({x:state[0], y:state[1], phi:state[2], vx:state[3], vy:state[4], omega:state[5],
                         d:control[0], delta:control[1], t:0.033})
        C = self.C.subs({x:state[0], y:state[1], phi:state[2], vx:state[3], vy:state[4], omega:state[5],
                         d:control[0], delta:control[1], t:0.033})
        '''
        args = (state[0], state[1], state[2], state[3], state[4], state[5], control[0], control[1], dt)
        A = self.A(*args)
        B = self.B(*args)
        C = self.C(*args)
        Anp = np.array(A).astype(np.float64)
        Bnp = np.array(B).astype(np.float64)
        Cnp = np.array(C).astype(np.float64).ravel()
        
        return Anp, Bnp, Cnp

    def step(self, state, control, dt=0.033):
        '''    
        f = self.f.subs({x:state[0], y:state[1], phi:state[2], vx:state[3], vy:state[4], omega:state[5],
                         d:control[0], delta:control[1]})
        '''
        f = self.f(state[0], state[1], state[2], state[3], state[4], state[5], control[0], control[1])
        fnp = np.array(f).astype(np.float64)
        next_state = state + fnp.T * dt

        return next_state
    
    '''
    Updates rollout states using rollout controls
    '''
    def rollout(self, state, rollout_controls, rollout_states, dt=0.033):
        rollout_states[:, 0] = state

        for i in range(rollout_states.shape[1] - 1):
            rollout_states[:, i + 1] = self.step(rollout_states[:, i], rollout_controls[:, i], dt=dt)

class TestKinematicModel(unittest.TestCase):
    def testStep(self):
        km = KinematicModel()
        state = np.zeros(3) #[0, 0, 0]
        velocity = 5.0
        steering_angle = 15 * math.pi / 180
        print(km.step(state, velocity, steering_angle, dt=2.0))

class TestDynamicModelSymPy(unittest.TestCase):
    def testStep(self):
        dmsp = DynamicModelSymPy()
        state = np.zeros(6)
        state[3] = 1
        control = np.zeros(2)
        control[0] = 1
        print(dmsp.step(state, control))

    def testLinearModel(self):
        dmsp = DynamicModelSymPy()
        state = np.zeros(6)
        state[3] = 1
        control = np.zeros(2)
        control[0] = 1
        A, B, C = dmsp.get_linear_model(state, control)
        print(A.shape)
        print(B.shape)
        print(C.shape)
        print(C)
