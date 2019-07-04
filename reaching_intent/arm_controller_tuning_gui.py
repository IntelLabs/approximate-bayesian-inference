#!/usr/bin/python3

###################################
# GENERIC IMPORTS
###################################
import sys
import time
from common.common import *
from samplers.CSamplerUniform import CSamplerUniform

###################################
# APPLICATION SPECIFIC IMPORTS (import from your application specific module)
###################################
import pybullet as p
from reaching_intent.generative_models.CGenerativeModelSimulator import CGenerativeModelSimulator
from reaching_intent.generative_models.CGenerativeModelSimulator import create_sim_params
from reaching_intent.generative_models.CReachingDataset import CReachingDataset
###################################

###################################
# GENERIC PARAMETERS (tune for each application)
###################################
dataset_path = "datasets/default.dat"

dataset_points = 1e4   # Number of data points that the dataset will contain

dataset_gen_batch = 1e3   # Max number of data points generated before saving to a file. Important to save batches
                          # when generating huge datasets to keep memory requirements bounded

if len(sys.argv) > 1:
    dataset_path = sys.argv[1]
###################################

###################################
# APPLICATION SPECIFIC PARAMETERS (add/remove/edit parameters to fit your implementation)
###################################
sim_viz           = True  # Visualize the generation process
sample_rate = 30

# Generative model parameter limits: start volume (x,y,z), end volume (x,y,z), controller(Kp,Ki,Kd,Krep)
param_limits_min = t_tensor([0.05, 0.5, 0.2,    0.25, -0.4, 0.2,     1, 0.005, 0, 0])
param_limits_max = t_tensor([-0.1, 0.2, -0.2,    0.9,   0.4, 0.21,    20, 0.01, 0, 0.05])

# Sampler to sample parameter values to generate trajectories
param_sampler = CSamplerUniform({"min": param_limits_min, "max": param_limits_max})
###################################


#################################################
# GENERIC CODE (dataset generation)
#################################################
# Get simulator parameters from the app specific import
simulator_params = create_sim_params(sim_viz=sim_viz, sample_rate=sample_rate)

# Simulator used to generate synthetic data
neSimulator = CGenerativeModelSimulator(simulator_params)

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
