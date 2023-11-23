import numpy as np
import math

import unittest

class KinematicModel:
    def __init__(self):
        self.lr = 0.147
        self.lf = 0.178
        self.L = self.lr + self.lf

    def step(self, state, velocity, steering_angle, dt=0.033):
        slip_angle = math.atan(self.lr / (self.lr + self.lf) * math.tan(steering_angle))

        next_state = np.zeros(state.shape)
        next_state[0] = state[0] + velocity * math.cos(state[2] + slip_angle) * dt
        next_state[1] = state[1] + velocity * math.sin(state[2] + slip_angle) * dt
        #next_state[2] = state[2] + velocity * math.sin(slip_angle) / self.lr * dt
        next_state[2] = state[2] + velocity * math.cos(slip_angle) * math.tan(steering_angle) \
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

    def rollout(self, state, control_sequence, steps):
        return states

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

class TestKinematicModel(unittest.TestCase):
    def testStep(self):
        km = KinematicModel()
        state = np.zeros(3) #[0, 0, 0]
        velocity = 5.0
        steering_angle = 15 * math.pi / 180
        print(km.step(state, velocity, steering_angle, dt=2.0))
