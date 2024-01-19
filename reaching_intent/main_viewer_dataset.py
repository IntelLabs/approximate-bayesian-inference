
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT

#!/usr/bin/python3
import time
import pybullet as p
import numpy as np

from reaching_intent.generative_models.CGenerativeModelSimulator import CGenerativeModelSimulator
from utils.draw import draw_trajectory
from utils.draw import draw_text
from reaching_intent.generative_models.CReachingDataset import CReachingDataset
from reaching_intent.generative_models.CGenerativeModelSimulator import create_sim_params
from reaching_intent.generative_models.CGenerativeModelSimulator import scene_with_ur5table

# Script configuration
dataset_path = "datasets/dataset10K_2D_ur5_96p.dat"
sample_rate = 30
sim_time = 3.2      # Duration of the simulated trajectories
noise_sigma = 0.001  # Noise added to the dataset trajectories
model_path = "pybullet_models/human_torso/model.urdf"

# Load simulator
simulator_params = create_sim_params(sim_time=sim_time, sample_rate=sample_rate)
scene_with_ur5table(simulator_params)
neSimulator = CGenerativeModelSimulator(simulator_params)  # Simulator used to generate synthetic data

dataset = CReachingDataset(filename=dataset_path, noise_sigma=noise_sigma,
                           dataset_sample_rate=sample_rate, output_sample_rate=sample_rate,
                           n_datapoints=100)

hold_on = False
ndims = 3

while p.isConnected(physicsClientId=neSimulator.sim_id):
    # Show a random trajectory
    idx = np.random.randint(0, len(dataset))
    dataset_traj = dataset.samples[idx][1]
    draw_trajectory(dataset_traj.view(-1, 3), color=[1, 1, 0], width=2, physicsClientId=neSimulator.sim_id,
                    draw_points=True)
    draw_text("  Trajectory id: " + str(idx), dataset_traj[0:3], visualizer=neSimulator.sim_id)

    keys = p.getKeyboardEvents(physicsClientId=neSimulator.sim_id)
    while ord('n') not in keys or keys[ord('n')] != p.KEY_WAS_RELEASED:
        keys = p.getKeyboardEvents(physicsClientId=neSimulator.sim_id)
        time.sleep(0.01)
        if ord('h') in keys and keys[ord('h')] != p.KEY_WAS_RELEASED:
            hold_on = not hold_on

    if not hold_on:
        p.removeAllUserDebugItems(physicsClientId=neSimulator.sim_id)
