
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT

from common.common import *
from common.CBaseObservationModel import CBaseObservationModel
from utils.misc import resample_trajectory


# TODO: Finalize refactoring of this observation model
class CObservationModelSimulator(CBaseObservationModel):
    def __init__(self, params):  #[model, viz, timestep, goal_threshold, ini, goal, sigma]
        raise NotImplementedError
        super(CObservationModelSimulator, self).__init__(params)
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
