
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT

#!/usr/bin/python3
"""
View the trajectories generated by a neural surrogate and compare them to a trajectory generated with the simulator.
"""

#################
# SYSTEM IMPORTS
#################
from common.common import *
import torch.nn
import pybullet as p
import time
from samplers.CSamplerUniform import CSamplerUniform

#######################################
# GENERIC IMPORTS (no need to edit)
#######################################
from neural_emulators.CGenerativeModelNeuralEmulator import CGenerativeModelNeuralEmulator
from reaching_intent.generative_models.CGenerativeModelSimulator import CGenerativeModelSimulator
from reaching_intent.generative_models.CGenerativeModelSimulator import create_sim_params
##############################


#######################################
# DEBUG DRAWING IMPORTS
#######################################
from utils.draw import draw_trajectory
from utils.draw import draw_trajectory_diff
from utils.draw import draw_point
from utils.draw import draw_text
##############################


###################################
# APPLICATION SPECIFIC PARAMETERS (add/remove/edit parameters to fit your implementation)
###################################
sample_rate = 30    # Sensor sampling rate
sim_time = 3.2      # Prediction time window in seconds
n_dims = 3          # Point dimensionality
n_points = int(sample_rate * sim_time)

# Generative model parameter limits: start volume (x,y,z), end volume (x,y,z), controller(Kp, Ki, Kd, iClamp, Krep)
param_limits_min = t_tensor([-0.06, 0.30, -0.10, 0.25, -0.4, 0.20, 5, 0.0, 0, 0, 90.0])
param_limits_max = t_tensor([-0.05, 0.31, -0.09, 0.90, 0.4, 0.60, 20, 0.01, 1.0, 2, 100.0])

# Select the parameters that are considered nuisance and the parameters that are considered interesting
latent_mask = t_tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]) == 1  # We are interested in the end position
nuisance_mask = t_tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]) == 0  # The rest of the parameters are considered nuisance

# Neural emulator path
nn_model_path = "pytorch_models/ne_fc3_10k3D_MSE_in11_out288.pt"
###################################


###############
# GENERIC CODE: No need to edit
###############
# Sampler to sample parameter values to generate trajectories
param_sampler = CSamplerUniform({"min": param_limits_min, "max": param_limits_max})

# Load the neural emulator generative model
neNEmulator = CGenerativeModelNeuralEmulator(nn_model_path)

# Load simulator for visualization purposes only
simulator_params = create_sim_params(sim_time=sim_time, sample_rate=sample_rate)
neSimulator = CGenerativeModelSimulator(simulator_params)

while p.isConnected(neSimulator.sim_id):
    # Sample a point in the goal space.
    params = param_sampler.sample(nsamples=1, params=None)
    z = params[:, latent_mask]
    n = params[:, nuisance_mask]

    # Generate a trajectory with the simulator
    traj_gt = neSimulator.generate(z, n)

    # Generate a trajectory with the emulator
    traj_gen = neNEmulator.generate(z.view(1, -1), n=None)[0]

    # Show both trajectories and the difference
    draw_trajectory(traj_gen.view(-1, 3), color=[1, 0, 0], width=2,
                    physicsClientId=neSimulator.sim_id, draw_points=True)
    draw_trajectory(traj_gt.view(-1, 3), color=[0, 1, 0], width=2,
                    physicsClientId=neSimulator.sim_id, draw_points=True)
    draw_trajectory_diff(traj_gen.view(-1, 3), traj_gt.view(-1, 3), color=[0, 0, 1], width=1,
                         physicsClientId=neSimulator.sim_id)
    draw_point(z[0], [0, 0, 1], size=0.05, width=5, physicsClientId=neSimulator.sim_id)
    loss, loss_terms = neNEmulator.model.criterion(traj_gen.view(-1, 3), traj_gt.view(-1, 3))
    draw_text("Test Gen: loss: %5.3f" %
              torch.sqrt(loss).mean().item(), traj_gen.view(-1, 3)[-1], neSimulator.sim_id, [1, 0, 0])
    draw_text("Test GT", traj_gt.view(-1, 3)[-1], neSimulator.sim_id, [0, 1, 0])

    # Wait for N or space keypress to generate next
    while p.isConnected(neSimulator.sim_id):
        keys = p.getKeyboardEvents(neSimulator.sim_id)
        if ord('n') in keys and keys[ord('n')] & p.KEY_WAS_TRIGGERED:
            break
        time.sleep(0.01)

    p.removeAllUserDebugItems()

