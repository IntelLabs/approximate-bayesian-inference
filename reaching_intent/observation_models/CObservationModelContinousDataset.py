from common.common import *
from reaching_intent.generative_models.CReachingDataset import CReachingDataset
from common.CBaseObservationModel import CBaseObservationModel


class CObservationModelContinousDataset(CBaseObservationModel):
    def __init__(self, params):
        super(CObservationModelContinousDataset, self).__init__(params)
        self.dataset = CReachingDataset(filename=params["dataset_path"], dataset_sample_rate=params["sample_rate"],
                                        output_sample_rate=params["sample_rate"], noise_sigma=params["sigma"])
        self.min_points = params["min_points"]
        self.dimensions = params["obs_dimensions"]
        self.obs_window = params["obs_window"]
        self.goal_window = params["goal_window"]
        self.traj_idx = 0
        self.idx = 0

    def __del__(self):
        pass

    def get_observation(self):
        idx_start = self.idx * self.dimensions
        idx_end = (self.idx + self.obs_window) * self.dimensions

        # Grab points from the current trajectory. Limited to its length
        self.traj = self.dataset.samples[self.traj_idx][1]
        self.traj = self.traj[:int(len(self.traj) / 2)]

        idx_end_clip = min(idx_end, len(self.traj))
        self.obs = self.traj[idx_start:idx_end_clip]

        # Grab points from the next trajectory if needed
        if idx_end_clip == len(self.traj):
            self.traj_next = self.dataset.samples[self.traj_idx+1][1]
            self.traj_next = self.traj_next[:int(len(self.traj_next) / 2)]
            obs_next = self.traj_next[0:idx_end-idx_end_clip]
            self.obs = torch.cat((self.obs, obs_next))

        self.step()
        return self.obs

    def step(self):
        self.idx = self.idx + 1

        # Check if we need to jump to the next traj
        if self.idx >= len(self.traj) / self.dimensions:
            self.traj_idx = self.traj_idx + 1
            self.idx = 0

    def is_ready(self):
        return self.traj_idx < len(self.dataset)

    def new_trajectory(self, idx=None):
        self.idx = 0

    def get_goal(self):
        traj_idx = self.traj_idx
        idx_end = self.goal_window * self.dimensions

        traj = self.traj[self.idx*self.dimensions:]
        if idx_end < len(traj):
            return traj[idx_end:idx_end + self.dimensions]

        idx_end = idx_end - len(traj)

        while idx_end >= len(traj):
            traj_idx = traj_idx + 1
            traj = self.dataset.samples[traj_idx][1]
            traj = traj[:int(len(traj) / 2)]
            if idx_end < len(traj):
                return traj[idx_end:idx_end + self.dimensions]

            idx_end = idx_end - len(traj)

        return traj[idx_end:idx_end+self.dimensions]
