import opengen as og
import casadi.casadi as cs 
import sys
import time
import numpy as np
import math

N = 50
T = 0.033

half_width = 2
car_width = 0.25
car_length = 0.4
lr = 0.147
lf = 0.178
car_diag = math.sqrt(car_width ** 2 + car_length ** 2)
security_distance = (half_width - car_diag /2 ) ** 2

v_max = 5
v_min = 0

delta_max = math.pi/6
delta_min = -delta_max

x = cs.SX.sym('x')
y = cs.SX.sym('y')
theta = cs.SX.sym('theta')
states = [x, y, theta]
n_states = len(states)

v = cs.SX.sym('v')
delta = cs.SX.sym('delta')
controls = [v, delta]
n_controls = len(controls)

'''
b = cs.atan(lr * cs.tan(lr * cs.tan(delta) / (lr + lf)))
f = cs.Function('f', [x, y, theta, v, delta],\
        [v * cs.cos(theta + b),
         v * cs.sin(theta + b),
         v * cs.cos(b) * cs.tan(delta) / (lr + lf)])
'''
f = cs.Function('f', [x, y, theta, v, delta],\
        [v * cs.cos(theta + cs.atan(lr * cs.tan(lr * cs.tan(delta) / (lr + lf)))),
         v * cs.sin(theta + cs.atan(lr * cs.tan(lr * cs.tan(delta) / (lr + lf)))),
         v * cs.cos(cs.atan(lr * cs.tan(lr * cs.tan(delta) / (lr + lf)))) * cs.tan(delta) / (lr + lf)])

U = cs.SX.sym('U', n_controls * N)
# P is for parameters. It contains initial state, previous control, target point, rollout projection to centerline
P = cs.SX.sym('P', n_states + n_controls + 2 + 2 * N)
X = cs.SX.sym('X', n_states * (N + 1))

X[0] = P[0]
X[1] = P[1]
X[2] = P[2]

obj = 0
f1 = []
preU1 = P[3]
preU2 = P[4]

for i in range(N):
    st1 = X[n_states * i]
    st2 = X[n_states * i + 1]
    st3 = X[n_states * i + 2]
    con1 = U[n_controls * i]
    con2 = U[n_controls * i + 1]
    obj = obj + 10 * (con1 - preU1) ** 2 + 10 * (con2 - preU2) ** 2 # control change cost
    f_value = f(st1, st2, st3, con1, con2)
    st_next1 = st1 + T * f_value[0]
    st_next2 = st2 + T * f_value[1]
    st_next3 = st3 + T * f_value[2]
    X[n_states * (i + 1)] = st_next1
    X[n_states * (i + 1) + 1] = st_next2
    X[n_states * (i + 1) + 2] = st_next3
    f1 = cs.vertcat(f1, (st1 - P[2 * (i + 1) + 5]) ** 2 + (st2 - P[2 * (i + 1) + 6]) ** 2)
    f1 = cs.vertcat(f1, con1 - preU1, con2 - preU2)
    preU1 = con1
    preU2 = con2
    #obj = obj + (((-cs.atan((st5+lf*st6)/st4)+con2)) - (-cs.atan((st5-lr*st6)/st4)))**2 # slip ratio difference cost

obj = obj + 10 * (st_next1 - P[5]) ** 2 + 10 * (st_next2 - P[6]) ** 2 # target point cost

umin = [v_min, delta_min] * N
umax = [v_max, delta_max] * N
bounds = og.constraints.Rectangle(umin, umax)

deltau1 = 0.5
deltau2 = 0.025 #0.025
bmin = [0, -deltau1, -deltau2] * N
bmax = [security_distance, deltau1, deltau2] * N
set_c = og.constraints.Rectangle(bmin, bmax)

problem = og.builder.Problem(U, P, obj).with_aug_lagrangian_constraints(f1, set_c).with_constraints(bounds)
build_config = og.config.BuildConfiguration()\
    .with_build_directory("optimizer")\
    .with_build_mode("release")\
    .with_build_python_bindings()
meta = og.config.OptimizerMeta()\
    .with_optimizer_name("nmpc_kinematic_optimizer")
solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-4)\
    .with_initial_tolerance(1e-4)\
    .with_max_outer_iterations(1)\
    .with_delta_tolerance(1e-1)\
    .with_penalty_weight_update_factor(10.0)\
    .with_max_duration_micros(30000)
builder = og.builder.OpEnOptimizerBuilder(problem, 
                                          meta,
                                          build_config, 
                                          solver_config) \
    .with_verbosity_level(1)
builder.build()
