
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT

import pybullet as p
import time
import torch
from generative_models import CGenerativeModelSimulator
from generative_models import CReachingNeuralEmulatorNN
from misc import draw_trajectory


def get_ik_solutions(model, eeffLink, goal, physicsClientId=0):
    pos = [0, 0, 0]
    rot = [0, 0, 0, 1]
    if len(goal) == 3:
        pos = goal
    elif len(goal) == 7:
        pos = goal[0:3]
        rot = goal[3:]
    else:
        return []

    return p.calculateInverseKinematics(model, eeffLink, pos, physicsClientId=physicsClientId)


def get_actuated_joint_idxs(model, physicsClientId=0):
    # Obtain the actuated joint indices
    actuated_joint_list = []
    for i in range(0, p.getNumJoints(model, physicsClientId=physicsClientId)):
        info = p.getJointInfo(model, i, physicsClientId=physicsClientId)  # Get all joint limits
        if not info[2] == p.JOINT_FIXED:
            actuated_joint_list.append(i)

    return actuated_joint_list


def set_model_position(goal, q_ini, model, eef_link, physicsClientId=0):
    # Obtain the actuated joint indices
    actuated_joint_list = get_actuated_joint_idxs(model, physicsClientId=physicsClientId)

    # Compute the joint position required to reach the specified goal in cartesian coordinates
    q = get_ik_solutions(model, eef_link, goal, physicsClientId=physicsClientId)

    # Move the joints to the desired position
    for i, j in enumerate(actuated_joint_list):
        p.resetJointState(model, j, q[i], physicsClientId=physicsClientId)


def play_trajectory(traj, sample_rate, ini_pos, model, eeflink, physicsClientId=0):
    # Obtain the actuated joint indices
    actuated_joint_list = get_actuated_joint_idxs(model, physicsClientId=physicsClientId)

    # Move the joints to the initial position
    for i, j in enumerate(actuated_joint_list):
        p.resetJointState(model, j, ini_pos[i], physicsClientId=physicsClientId)

    for point in traj:
        set_model_position(point, ini_pos, model, eeflink, physicsClientId)
        time.sleep(1/sample_rate)


def pybullet_wait_for_key(key, physicsClientId=0):

    keys = p.getKeyboardEvents(physicsClientId=physicsClientId)

    while ord(key) not in keys or keys[ord(key)] != p.KEY_WAS_RELEASED:

        keys = p.getKeyboardEvents(physicsClientId=physicsClientId)

        time.sleep(0.1)


def parse_nn_trajectory(traj_nn):
    nn_traj_cart_points = []

    for i in range(0, len(traj_nn), 3):
        nn_traj_cart_points.append((traj_nn[i].item(), traj_nn[i + 1].item(), traj_nn[i + 2].item()))

    return nn_traj_cart_points


if __name__ == "__main__":

    input_params = [
        [.4, .5, .3, 2],  [.4, .5, .3, 10],
        [.4, -.5, .3, 2], [.4, -.5, .3, 10],
        [.4, 0, .3, 2],   [.4, 0, .3, 10],
        [.2, .5, .3, 2],  [.2, .5, .3, 10],
        [.2, -.5, .3, 2], [.2, -.5, .3, 10],
        [.2, 0, .3, 2],   [.2, 0, .3, 10],
        [.6, .5, .3, 2],  [.6, .5, .3, 10],
        [.6, -.5, .3, 2], [.6, -.5, .3, 10],
        [.6, 0, .3, 2],   [.6, 0, .3, 10],
        [.4, .5, .6, 2],  [.4, .5, .6, 10],
        [.4, -.5, .6, 2], [.4, -.5, .6, 10],
        [.4, 0, .6, 2],   [.4, 0, .6, 10],
        [.2, .5, .6, 2],  [.2, .5, .6, 10],
        [.2, -.5, .6, 2], [.2, -.5, .6, 10],
        [.2, 0, .6, 2],   [.2, 0, .6, 10],
        [.6, .5, .6, 2],  [.6, .5, .6, 10],
        [.6, -.5, .6, 2], [.6, -.5, .6, 10],
        [.6, 0, .6, 2],   [.6, 0, .6, 10]
    ]

    urdf_model = "models/model.urdf"
    nn_model_path = "models/test1k_MLE.pt"
    timestep = 0.1
    sim_time = 5.0

    ieGeneratorSim = CGenerativeModelSimulator([urdf_model, False, timestep, sim_time])

    ieVisualizationSim = CGenerativeModelSimulator([urdf_model, True, timestep, sim_time])

    try:
        neNEmulator = torch.load(nn_model_path)
    except FileNotFoundError as e:
        print(e)
        exit(-1)

    ini_pos = [0] * p.getNumJoints(ieVisualizationSim.model_id, physicsClientId=ieVisualizationSim.sim_id)

    for params in input_params:
        p.addUserDebugText("K=%.2f" % params[3], [.1, 0, .5], physicsClientId=ieVisualizationSim.sim_id)

        # Obtain simulated trajectory
        sim_traj = ieGeneratorSim.generate(ini_pos + params)

        # Obtain emulated trajectory
        ne_output = neNEmulator(torch.DoubleTensor(params).to(neNEmulator.device).view(1, 4))
        emu_traj = parse_nn_trajectory(ne_output[0, 0:neNEmulator.output_dim])

        # Play the simulated trajectory
        play_trajectory(sim_traj, 1 / timestep, ini_pos, ieVisualizationSim.model_id, ieVisualizationSim.eef_link, ieVisualizationSim.sim_id)

        # Display it in green
        draw_trajectory(sim_traj, [0, 1, 0], 3, physicsClientId=ieVisualizationSim.sim_id)

        # Play the emulated trajectory
        play_trajectory(emu_traj, 1 / timestep, ini_pos, ieVisualizationSim.model_id, ieVisualizationSim.eef_link, ieVisualizationSim.sim_id)

        # Display it in red
        draw_trajectory(emu_traj, [1, 0, 0], 3, physicsClientId=ieVisualizationSim.sim_id)

        pybullet_wait_for_key('n', ieVisualizationSim.sim_id)

        # Clean the trajectories
        p.removeAllUserDebugItems(physicsClientId=ieVisualizationSim.sim_id)
