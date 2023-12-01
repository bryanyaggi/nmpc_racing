import math
import numpy as np

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

def get_closest_point_on_centerline(x, y, center_x, center_y):
    dx = (x - np.array(center_x)) ** 2
    dy = (y - np.array(center_y)) ** 2
    dr = dx + dy

    index = np.argmin(dr)
    
    return index

'''
Returns equally spaced points along centerline
'''
def sample_centerline(start_x, start_y, center_x, center_y, points_in=91, points_out=51):
    # Get start and end indices
    start_i = get_closest_point_on_centerline(start_x, start_y, center_x, center_y)
    if start_i + points_in < len(center_x):
        segment_center_x = center_x[start_i:start_i+points_in]
        segment_center_y = center_y[start_i:start_i+points_in]
    else:
        end_i = points_in - (len(center_x) - 1 - start_i)
        segment_center_x = center_x[start_i:] + center_x[:end_i]
        segment_center_y = center_y[start_i:] + center_y[:end_i]

    # Get path segment
    segment_center_x = np.array(segment_center_x)
    segment_center_y = np.array(segment_center_y)
    dx, dy = segment_center_x[1:] - segment_center_x[:-1], segment_center_y[1:] - segment_center_y[:-1]
    ds = np.array((0, *np.sqrt(dx**2 + dy**2))) # distance along path between points
    s = np.cumsum(ds) # distance from start

    # Interpolate
    #spacing = s[-1] / (points_out - 1)
    #x = np.interp(np.arange(0, s[-1] + spacing, spacing), s, segment_center_x)
    #y = np.interp(np.arange(0, s[-1] + spacing, spacing), s, segment_center_y)
    x = np.interp(np.linspace(0, s[-1], points_out), s, segment_center_x)
    y = np.interp(np.linspace(0, s[-1], points_out), s, segment_center_y)

    return x, y

def get_path_yaw(path_x, path_y):
    dx, dy = path_x[1:] - path_x[:-1], path_y[1:] - path_y[:-1]
    yaw = np.arctan2(dy, dx)

    return yaw

class Test(unittest.TestCase):
    def setUp(self):
        import csv

        csv_file = np.genfromtxt('/home/ubuntu/project/nmpc_racing/optimization/Map_track3/center_x_track3.csv', 
                          delimiter=',', dtype=float)
        self.center_x = csv_file[:].tolist()
        csv_file = np.genfromtxt('/home/ubuntu/project/nmpc_racing/optimization/Map_track3/center_y_track3.csv', 
                          delimiter=',', dtype=float)
        self.center_y = csv_file[:].tolist()

    def testSampleCenterline(self):
        x, y = 0, 0
        path_x, path_y = sample_centerline(x, y, self.center_x, self.center_y)
        print(path_x)
        print(path_y)
        print(len(path_x))
        
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(self.center_x, self.center_y, color='black')
        ax.plot(path_x, path_y, color='blue')
        plt.show()

    def testGetPathYaw(self):
        x, y = 0, 0
        path_x, path_y = sample_centerline(0, 0, self.center_x, self.center_y)
        print(path_x)
        print(path_y)

        yaw = get_path_yaw(path_x, path_y)
        print(yaw)
        
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(path_x, path_y)
        ax.axis('equal')
        plt.show()
