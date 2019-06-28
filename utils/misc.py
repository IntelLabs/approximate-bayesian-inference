import numpy as np
import torch
from neural_emulators.common import *

def resample_trajectory(traj, old_sample_rate, new_sample_rate, ndims=3):

        resampled = t_tensor([])

        n_points = (new_sample_rate / old_sample_rate) * (len(traj)/ndims)

        traj_len = len(traj) / ndims
        traj_ini = 0
        traj_end = traj_len - 1
        traj_incr = traj_len / n_points
        for s in np.arange(traj_ini, traj_end, traj_incr):
            resampled = torch.cat((resampled, traj[int(s)*ndims:int(s)*ndims+ndims]))

        if len(resampled) > n_points * ndims:
            resampled = resampled[0:int(n_points) * ndims]

        return resampled

def to_indices(space_min, resolution, x, y, z):
    i = int((x - space_min[0]) / resolution)
    j = int((y - space_min[1]) / resolution)
    k = int((z - space_min[2]) / resolution)
    return [i, j, k]


def to_cartesian(space_min, resolution, i, j, k):
    x = float(i) * resolution + space_min[0]
    y = float(j) * resolution + space_min[1]
    z = float(k) * resolution + space_min[2]
    res = torch.DoubleTensor([x, y, z])
    return res
