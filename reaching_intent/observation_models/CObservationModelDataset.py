
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT

from common.common import *
from reaching_intent.generative_models.CReachingDataset import CReachingDataset
from common.CBaseObservationModel import CBaseObservationModel


class CObservationModelDataset(CBaseObservationModel):
    def __init__(self, params):
        super(CObservationModelDataset, self).__init__(params)
        self.n_points = params["sample_rate"] * params["episode_time"]

        self.dataset = CReachingDataset(filename=params["dataset_path"], dataset_sample_rate=params["sample_rate"],
                                        output_sample_rate=params["sample_rate"], noise_sigma=params["sigma"],
                                        n_datapoints=params["num_trajs"], traj_duration=params["episode_time"])
        self.min_points = params["min_points"]
        self.dimensions = params["obs_dimensions"]
        self.new_trajectory()

    def get_ground_truth(self):
        return self.goal

    def get_ground_truth_trajectory(self):
        return self.traj

    def get_observation(self):
        self.obs = self.traj[0:self.idx*self.dimensions]
        self.step()
        return self.obs

    def step(self):
        self.idx = self.idx + 1

    def is_ready(self):
        return self.idx < len(self.traj) / self.dimensions

    def new_trajectory(self, idx=None):
        if idx is None:
            self.traj_idx = int((torch.rand(1) * len(self.dataset)).item())
        else:
            self.traj_idx = idx
        print("Observation trajectory. Selected traj idx: %d" % self.traj_idx)
        self.traj = self.dataset.y_samples[self.traj_idx]
        self.idx = self.min_points
        self.goal = self.dataset.x_samples[self.traj_idx]
