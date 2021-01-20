#!/usr/bin/python3
import time
import pybullet as p
import numpy as np

from reaching_intent.generative_models.CGenerativeModelSimulator import CGenerativeModelSimulator
from utils.draw import draw_trajectory
from utils.draw import draw_text
from reaching_intent.generative_models.CReachingDataset import CReachingDataset


# Script configuration
dataset_path = "datasets/default.dat"
sample_rate = 30
noise_sigma = 0.005  # Noise added to the dataset trajectories
model_path = "pybullet_models/human_torso/model.urdf"


# Load simulator
simulator_params = dict()
simulator_params["robot_model_path"] = model_path
simulator_params["visualization"] = True
simulator_params["timestep"] = 0.01
simulator_params["episode_time"] = 3.0
simulator_params["sample_rate"] = sample_rate
simulator_params["sim_id"] = 0
simulator_params["robot_controller"] = None
simulator_objects = dict()
simulator_objects["path"] = ["pybullet_models/table/table.urdf"]
simulator_objects["pose"] = [[0.6, 0, -0.65]]
simulator_objects["static"] = [True]
simulator_params["objects"] = simulator_objects
neSimulator = CGenerativeModelSimulator(simulator_params)    # Simulator used to generate synthetic data

dataset = CReachingDataset(filename=dataset_path, noise_sigma=noise_sigma,
                           dataset_sample_rate=sample_rate, output_sample_rate=sample_rate)

hold_on = False
ndims = 3

while p.isConnected(physicsClientId=neSimulator.sim_id):
    # Show a random trajectory
    idx = np.random.randint(0, len(dataset))
    dataset_traj = dataset.samples[idx][1]
    draw_trajectory(dataset_traj.view(-1, 3), color=[1, 1, 0], width=2, physicsClientId=neSimulator.sim_id, draw_points=True)
    draw_text("  Trajectory id: " + str(idx), dataset_traj[0:3], visualizer=neSimulator.sim_id)

    keys = p.getKeyboardEvents(physicsClientId=neSimulator.sim_id)
    while ord('n') not in keys or keys[ord('n')] != p.KEY_WAS_RELEASED:
        keys = p.getKeyboardEvents(physicsClientId=neSimulator.sim_id)
        time.sleep(0.01)
        if ord('h') in keys and keys[ord('h')] != p.KEY_WAS_RELEASED:
            hold_on = not hold_on

    if not hold_on:
        p.removeAllUserDebugItems(physicsClientId=neSimulator.sim_id)
