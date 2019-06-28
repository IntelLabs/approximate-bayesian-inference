#!/usr/bin/python3

###################################
# GENERIC IMPORTS
###################################
import sys

import time

from neural_emulators.common import *

from neural_emulators.samplers import CUniformSampler

###################################
# APPLICATION SPECIFIC IMPORTS (import from your application specific module)
###################################
import pybullet as p

from reaching_intent_estimation.generative_models import CGenerativeModelSimulator

from reaching_intent_estimation.generative_models import CReachingDataset

from manipulator_planning_control import pybullet_utils as pb
from manipulator_planning_control import pybullet_controller as pbc
###################################

###################################
# GENERIC PARAMETERS (tune for each application)
###################################
if len(sys.argv) > 1:
    dataset_path = sys.argv[1]
else:
    dataset_path      = "datasets/default.dat"

dataset_points    = 1e4   # Number of data points that the dataset will contain

dataset_gen_batch = 1e3   # Max number of data points generated before saving to a file. Important to save batches
                          # when generating huge datasets to keep memory requirements bounded
###################################

###################################
# APPLICATION SPECIFIC PARAMETERS (add/remove/edit parameters to fit your implementation)
###################################
sim_viz           = True  # Visualize the generation process

sim_timestep      = 0.01  # Simulator time step (1/sample_rate)

sim_time = 5.0

model_path       = "models/human_torso/model.urdf"

#                            start               end                  controller Kp,Ki,Kd,Krep
# param_limits_min = t_tensor([0.15, -0.4, 0.2,    0.25, -0.4, 0.2,     1, 0, 0.001, 0])   # Goal volume lower bound
#
# param_limits_max = t_tensor([0.9,   0.4, 0.4,    0.9,   0.4, 0.21,    10, 0, 0.001, 1])   # Goal volume upper bound

#                            start               end                  controller Kp,Ki,Kd,Krep
param_limits_min = t_tensor([0.05, 0.5, 0.2,    0.25, -0.4, 0.2,     1, 0.005, 0, 0])   # Goal volume lower bound

param_limits_max = t_tensor([-0.1, 0.2, -0.2,    0.9,   0.4, 0.21,    20, 0.01, 0, 0.05])   # Goal volume upper bound

param_sampler = CUniformSampler(param_limits_min, param_limits_max)

sample_rate = 30

simulator_params = dict()
simulator_params["robot_model_path"] = model_path
simulator_params["visualization"] = sim_viz
simulator_params["timestep"] = sim_timestep
simulator_params["episode_time"] = sim_time
simulator_params["sample_rate"] = sample_rate
simulator_params["sim_id"] = 0
simulator_objects = dict()
simulator_objects["path"] = ["models/table/table.urdf"]
simulator_objects["pose"] = [[0.6, 0, -0.65]]
simulator_objects["static"] = [True]
simulator_params["objects"] = simulator_objects

# Controller parameters
Kp = 10
Kd = 0
Ki = 0.01
iClamp = 25.0
Krep = 0.1
PIDcontroller = pbc.CPIDController(Kp=Kp, Kd=Kd, Ki=Ki, iClamp=iClamp)  # Controller to be used
controller = pbc.CPotentialFieldController(Krep=Krep, ctrl=PIDcontroller)
simulator_params["robot_controller"] = controller
###################################


#################################################
# GENERIC CODE (dataset generation)
#################################################
neSimulator = CGenerativeModelSimulator(simulator_params)    # Simulator used to generate synthetic data

# Set the resting position for each joint
joint_indices = pb.get_actuable_joint_indices(neSimulator.model_id, neSimulator.sim_id)
joint_rest_positions = torch.zeros(len(joint_indices))
joint_rest_positions[5] = -1.04  # Elbow at 60 deg
joint_rest_positions[6] = 1.57  # Palm down
neSimulator.joint_rest_positions = joint_rest_positions
neSimulator.reset()

# Check if the dataset is completely generated. Generate it otherwise
print("Load/generate dataset...")
dataset = CReachingDataset(filename=dataset_path, dataset_sample_rate=sample_rate, output_sample_rate=sample_rate)
dataset_size = len(dataset)

if sim_viz:
    controller_params = dict()
    controller_params["Kp"] = p.addUserDebugParameter(paramName="Kp", rangeMin=0.0, rangeMax=100.0, startValue=Kp, physicsClientId=neSimulator.sim_id)
    controller_params["Kd"] = p.addUserDebugParameter(paramName="Kd", rangeMin=0.0, rangeMax=10.0, startValue=Kd, physicsClientId=neSimulator.sim_id)
    controller_params["Ki"] = p.addUserDebugParameter(paramName="Ki", rangeMin=0.0, rangeMax=10.0, startValue=Ki, physicsClientId=neSimulator.sim_id)
    controller_params["iClamp"] = p.addUserDebugParameter(paramName="iClamp", rangeMin=0.0, rangeMax=100.0, startValue=iClamp, physicsClientId=neSimulator.sim_id)
    controller_params["Krep"] = p.addUserDebugParameter(paramName="Krep", rangeMin=0.0, rangeMax=10.0, startValue=Krep, physicsClientId=neSimulator.sim_id)

while dataset_size < dataset_points:
    print("Loaded dataset has %d data points. Generating more..." % dataset_size)
    dataset.samples.clear()
    if sim_viz:
        Kp = p.readUserDebugParameter(controller_params["Kp"], physicsClientId=neSimulator.sim_id)
        Kd = p.readUserDebugParameter(controller_params["Kd"], physicsClientId=neSimulator.sim_id)
        Ki = p.readUserDebugParameter(controller_params["Ki"], physicsClientId=neSimulator.sim_id)
        iClamp = p.readUserDebugParameter(controller_params["iClamp"], physicsClientId=neSimulator.sim_id)
        Krep = p.readUserDebugParameter(controller_params["Krep"], physicsClientId=neSimulator.sim_id)
        neSimulator.controller.Krep = Krep
        neSimulator.controller.ctrl.Kp = Kp
        neSimulator.controller.ctrl.Kd = Kd
        neSimulator.controller.ctrl.Ki = Ki
        neSimulator.controller.ctrl.iClamp = iClamp

    for i in range(int(dataset_gen_batch)):
        t_ini = time.time()
        params = param_sampler.sample(None)
        if len(dataset) > 0:
            params[0:3] = dataset.samples[-1][1][int(len(dataset.samples[-1][1]) / 2) - 3:int(len(dataset.samples[-1][1]) / 2)] # Set the next starting position as the last motion position
        generated = neSimulator.generate(params.view(1,-1))
        dataset.samples.append([params, generated.view(-1)])
        print("Traj %d / %d. Time %2.4f" % ( len(dataset), dataset_points, time.time() - t_ini) )

    dataset.dataset_save(dataset_path)
    dataset_size = dataset_size + len(dataset)


print("Generated dataset with %d data points." % dataset_size)
