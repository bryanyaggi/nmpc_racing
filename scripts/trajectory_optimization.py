import cvxpy as cp
import unittest

def cvxopt(model, state, prev_control, target_point, operating_points, horizon)
    x = cp.Variable((4, horizon + 1))
    u = cp.Variable((2, horizon))

    Q1 = np.eye(2) * 10
    Q2 = np.eye(2) * 10

    cost = 0
    constraints = []
    for i in range(H):
        constraints += [x[:, t + 1] == As[t] @ x[:, t] + Bs[t] @ u[:, t] + Cs[t]] # dynamics

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

def Test(unittest.TestCase):
    def test(self):
        pass

if __name__ == '__main__':
    unittest.main()
