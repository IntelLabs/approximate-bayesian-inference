from common.common import *
from reaching_intent.generative_models.CReachingDataset import CReachingDataset
from common.CBaseObservationModel import CBaseObservationModel


class CObservationModelDataset(CBaseObservationModel):
    def __init__(self, params):
        super(CObservationModelDataset, self).__init__(params)
        self.n_points = params["sample_rate"] * params["episode_time"]

        self.dataset = CReachingDataset(filename=params["dataset_path"], dataset_sample_rate=params["sample_rate"],
                                        output_sample_rate=params["sample_rate"], noise_sigma=params["sigma"])
        self.min_points = params["min_points"]
        self.dimensions = params["obs_dimensions"]
        self.traj_idx = int((torch.rand(1) * len(self.dataset)).item())
        self.traj = self.dataset.y_samples[self.traj_idx]
        # self.traj = self.traj[:int(len(self.traj) / 2)]
        # self.traj_cov = self.traj[int(len(self.traj)/2):]
        self.idx = self.min_points
        self.incr = int(len(self.traj) / self.n_points)
        self.goal = self.traj[-self.dimensions:]

    def __del__(self):
        pass

    def get_goal(self):
        return self.traj[-self.dimensions:]

    def get_observation(self):
        self.obs = self.traj[0:self.idx*self.dimensions]
        self.step()
        # old_sample_rate = (len(self.traj)/self.dimensions) / self.tra
        # res = resample_trajectory(self.obs, old_sample_rate, self.sample_rate)
        return self.obs

    def step(self):
        self.idx = self.idx + 1
        # if self.traj is not None and len(self.traj) > self.idx:
        #     ik_solution = get_ik_jac_pinv_ns(self.model_id, self.eef_link, self.traj[self.idx:self.idx+self.dimensions])
        #     set_joint_angles(self.model_id, ik_solution, self.sim_id)

    def is_ready(self):
        return self.idx < len(self.traj) / self.dimensions

    def new_trajectory(self, idx=None):
        if idx is None:
            self.traj_idx = int((torch.rand(1) * len(self.dataset)).item())
        else:
            self.traj_idx = idx
        self.traj = self.dataset.y_samples[self.traj_idx]
        # self.traj = self.traj[:int(len(self.traj ) / 2)]
        # self.traj_cov = self.traj[int(len(self.traj )/2):]
        self.idx = self.dimensions * self.min_points
        self.incr = int(len(self.traj) / self.n_points)
        self.goal = self.traj[-self.dimensions:]
