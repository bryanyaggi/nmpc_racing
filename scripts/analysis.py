import csv

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap

import numpy as np
import math

from track_utils import get_track_data
from controllers import NMPC, NMPCKinematic
from models import DynamicModel, KinematicModel

class Track:
    def __init__(self, track_number='1'):
        self.center_x, self.center_y, self.bound_x1, self.bound_y1, self.bound_x2, self.bound_y2 = \
                get_track_data(track_number)

    def plot(self, ax):
        ax.plot(self.center_x, self.center_y, color='black', linestyle='dashed')
        ax.plot(self.bound_x1, self.bound_y1, color='black')
        ax.plot(self.bound_x2, self.bound_y2, color='black')

class Trajectory:
    def __init__(self, filename, controller_type='nmpc'):
        if controller_type == 'nmpc':
            self.model = DynamicModel()
        elif controller_type == 'nmpck':
            self.model = KinematicModel()
        self.controller_type = controller_type
        self.filename = filename
    
    def get_trajectory(self):
        self.data = np.genfromtxt(self.filename, delimiter=',', dtype=float)

    def get_lap_indices(self, laps=3):
        indices = [0]
        start = self.data[0, 1:3]
        while(len(indices) < laps):
            imin = indices[-1] + 100
            imax = min(indices[-1] + 700, self.data.shape[0])
            index = (np.linalg.norm(self.data[imin:imax, 1:3] - start, axis=1)).argmin() + imin
            indices.append(index)
            print(indices)
        print(indices)

        self.lap_indices = indices

    def get_lap_times(self):
        lap_times = []
        for i in range(1, len(self.lap_indices)):
            start_index = self.lap_indices[i - 1]
            end_index = self.lap_indices[i]
            lap_times.append(self.data[end_index, 0] - self.data[start_index, 0])
        print(lap_times)

        self.lap_times = lap_times

    def get_lateral_acceleration(self):
        self.lateral_acceleration = np.zeros(self.data.shape[0])
        
        for i in range(self.data.shape[0]):
            if self.controller_type == 'nmpc':
                self.lateral_acceleration[i] = abs(self.model.lateral_acceleration(self.data[i, 1:7], self.data[i,
                    7:9]))
            elif self.controller_type == 'nmpck':
                self.lateral_acceleration[i] = abs(self.model.lateral_acceleration(self.data[i, 4:6]))

        print(self.lateral_acceleration.max())

    def plot(self, fig, ax):
        end = self.lap_indices[2] + 1
        x = self.data[:end, 1]
        y = self.data[:end, 2]
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        v = self.data[:, 4]
        norm = plt.Normalize(v.min(), v.max())
        #norm = plt.Normalize(self.lateral_acceleration.min(), self.lateral_acceleration.max())
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(v)
        #lc.set_array(self.lateral_acceleration)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        cbar = fig.colorbar(line, ax=ax, pad=0)
        cbar.set_label('Velocity [m/s]')

def plot_data(filename, track_number, controller_type):
    fig, ax = plt.subplots()

    track = Track(track_number)
    track.plot(ax)

    trajectory = Trajectory(filename, controller_type)
    trajectory.get_trajectory()
    trajectory.get_lap_indices()
    trajectory.get_lap_times()
    trajectory.get_lateral_acceleration()
    trajectory.plot(fig, ax)

    plt.show()

def plot_track(fig, ax, filename, track_number):
    track = Track(track_number)
    track.plot(ax)

    trajectory = Trajectory(filename)
    trajectory.get_trajectory()
    trajectory.get_lap_indices()
    trajectory.plot(fig, ax)

    ax.axis('equal')
    ax.set_title('Track ' + track_number)

def plot_tracks(filenames):
    aspect_ratio = 4
    scale = 6
    fig, axs = plt.subplots(1, len(filenames), squeeze=False, figsize=(aspect_ratio * scale, scale))
    
    SMALL_SIZE = 12
    LARGE_SIZE = 18
    plt.rc('font', size=LARGE_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LARGE_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LARGE_SIZE)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=LARGE_SIZE)   # fontsize of the figure title

    for i in range(len(filenames)):
        plot_track(fig, axs[0, i], filenames[i], str(i + 1))

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    #plot_data('/home/ubuntu/project/nmpc_racing/data/nmpc_1.csv', '1', 'nmpc')
    #plot_data('/home/ubuntu/project/nmpc_racing/data/nmpck_1.csv', '1', 'nmpck')
    plot_tracks(['/home/ubuntu/project/nmpc_racing/data/nmpc_1.csv',
                 '/home/ubuntu/project/nmpc_racing/data/nmpc_2.csv',
                 '/home/ubuntu/project/nmpc_racing/data/nmpc_3.csv'])
