import sys, os
sys.path.insert(0, os.path.abspath('..'))

import torch
import pybullet as p
from manipulator_planning_control.pybullet_utils import get_ik_jac_pinv_ns
from manipulator_planning_control.pybullet_utils import set_joint_angles
from reaching_intent_estimation.generative_models import CGenerativeModelSimulator
from neural_emulators_code.utils.misc import resample_trajectory
from reaching_intent_estimation.generative_models import CReachingDataset
from neural_emulators.observation_models import CObservationModel

# TODO: Fix this observation model. With the new controller implementation
class CObservationModelSimulator(CGenerativeModelSimulator):
    def __init__(self, params):  #[model, viz, timestep, goal_threshold, ini, goal, sigma]
        super(CObservationModelSimulator,self).__init__(params)
        self.ini = params[3]
        self.goal = params[4]
        self.sigma = params[5]
        self.goal_threshold = params[6]
        self.time_window = params[7]
        self.sample_rate = params[9]
        self.obs = torch.DoubleTensor()
        self.timestamp = torch.DoubleTensor([0])
        self.noise = torch.distributions.normal.Normal(0, self.sigma)
        self.njoints = p.getNumJoints(self.model_id, physicsClientId=self.sim_id)

        # Move the joints to the starting state or zero if the initial state is empty
        if len(self.ini) != self.njoints:
            self.ini = [0] * self.njoints

        for i in range(0, p.getNumJoints(self.model_id, physicsClientId=self.sim_id)):
            p.resetJointState(self.model_id, i, self.ini[i], physicsClientId=self.sim_id)

    def disconnect(self):
        p.disconnect(physicsClientId=self.sim_id)

    def get_observation(self, use_timestamp=False):

        # Populate the observation with a minimum of time_window samples
        while len(self.obs)-1 < self.time_window:
            if use_timestamp:
                self.obs = torch.cat((self.obs, self.get_observation_cart()))
            else:
                self.obs = torch.cat((self.obs, self.get_observation_cart()[0:3]))
            self.step()

        # Add current observation and return subsampled observation
        if use_timestamp:
            self.obs = torch.cat((self.obs, self.get_observation_cart()))
        else:
            self.obs = torch.cat((self.obs, self.get_observation_cart()[0:3]))
        self.step()

        return resample_trajectory(self.obs, 1/self.timestep, self.sample_rate)

    def get_observation_cart(self):
        link_state = p.getLinkState(self.model_id, self.eef_link, 0, 1, physicsClientId=self.sim_id)
        link_state = torch.DoubleTensor(link_state[0])
        link_state = link_state + self.noise.sample(link_state.size()).double()
        res = torch.cat((link_state,  self.timestamp))
        return res

    def get_observation_joints(self):
        # Obtain current joint positions
        joint_state = p.getJointStates(self.model_id, range(0, p.getNumJoints(self.model_id, physicsClientId=self.sim_id)), physicsClientId=self.sim_id)
        joint_pos = torch.zeros(p.getNumJoints(self.model_id, physicsClientId=self.sim_id)).double()
        for i in range(0, p.getNumJoints(self.model_id, physicsClientId=self.sim_id)):
            joint_pos[i] = joint_state[i][0]

        return joint_pos + self.noise.sample(joint_pos.size()).double()

    def step(self):
        self.step_plan_potential_field(self.model_id, self.goal, self.eef_link, self.obstacles, physicsClientId=self.sim_id)
        self.timestamp = self.timestamp + self.timestep

    def is_ready(self):
        link_state = p.getLinkState(self.model_id, self.eef_link, 0, 1, physicsClientId=self.sim_id)
        error = self.goal - torch.DoubleTensor(link_state[0]).to(self.goal.device)
        error2 = torch.sum(error * error)

        return torch.sqrt(error2) > self.goal_threshold

    def gradient(self):
        raise NotImplementedError


class CObservationModelDataset(CObservationModel):
    def __init__(self, params):
        super(CObservationModelDataset, self).__init__(params)
        self.n_points = params["sample_rate"] * params["episode_time"]

        self.dataset = CReachingDataset(filename=params["dataset_path"], dataset_sample_rate=params["sample_rate"],
                                        output_sample_rate=params["sample_rate"], noise_sigma=params["sigma"])
        self.min_points = params["min_points"]
        self.dimensions = params["obs_dimensions"]
        self.traj_idx = int((torch.rand(1) * len(self.dataset)).item())
        self.traj = self.dataset.samples[self.traj_idx][1]
        self.traj = self.traj[:int(len(self.traj ) / 2)]
        self.traj_cov = self.traj[int(len(self.traj )/2):]
        self.idx = self.min_points
        self.incr = int(len(self.traj) / self.n_points)
        self.goal = self.dataset.samples[self.traj_idx][1][-self.dimensions:]

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
        self.traj = self.dataset.samples[self.traj_idx][1]
        self.traj = self.traj[:int(len(self.traj ) / 2)]
        self.traj_cov = self.traj[int(len(self.traj )/2):]
        self.idx = self.dimensions * self.min_points
        self.incr = int(len(self.traj) / self.n_points)
        self.goal = self.dataset.samples[self.traj_idx][1][-self.dimensions:]


class CObservationModelContinousDataset(CObservationModel):
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
        self.traj = self.traj[:int(len(self.traj ) / 2)]

        idx_end_clip = min(idx_end, len(self.traj))
        self.obs = self.traj[idx_start:idx_end_clip]

        # Grab points from the next trajectory if needed
        if idx_end_clip == len(self.traj):
            self.traj_next = self.dataset.samples[self.traj_idx+1][1]
            self.traj_next = self.traj_next[:int(len(self.traj_next) / 2)]
            obs_next = self.traj_next[0:idx_end-idx_end_clip]
            self.obs = torch.cat((self.obs,obs_next))

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
