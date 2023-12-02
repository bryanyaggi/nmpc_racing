import csv

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap

import numpy as np
import math

from track_utils import get_track_data
from controllers import NMPC, NMPCKinematic

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
            self.controller = NMPC()
        self.controller_type = controller_type
        self.filename = filename
    
    def get_trajectory(self):
        self.data = np.genfromtxt(self.filename, delimiter=',', dtype=float)

    def get_lap_indices(self, laps=4):
        indices = [0]
        start = self.data[0, 1:3]
        while(len(indices) < laps):
            imin = indices[-1] + 100
            imax = min(indices[-1] + 700, self.data.shape[0])
            index = (np.linalg.norm(self.data[imin:imax, 1:3] - start, axis=1)).argmin() + imin
            indices.append(index)
        print(indices)

        self.lap_indices = indices

    def plot(self, fig, ax):
        end = self.lap_indices[2] + 1
        x = self.data[:end, 1]
        y = self.data[:end, 2]
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        if self.controller_type == 'nmpc':
            v = self.data[:, 4]

        norm = plt.Normalize(v.min(), v.max())
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(v)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        fig.colorbar(line, ax=ax)

def plot_data(filename, track_number, controller_type):
    fig, ax = plt.subplots()

    track = Track(track_number)
    track.plot(ax)

    trajectory = Trajectory(filename, controller_type)
    trajectory.get_trajectory()
    trajectory.get_lap_indices()
    trajectory.plot(fig, ax)

    plt.show()

if __name__ == '__main__':
    plot_data('/home/ubuntu/project/nmpc_racing/data/nmpc_1.csv', '1', 'nmpc')
