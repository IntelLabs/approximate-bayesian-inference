
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT

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
from reaching_intent.generative_models.CReachingDataset import CReachingDataset
from reaching_intent.generative_models.CGenerativeModelSimulator import CGenerativeModelSimulator
from reaching_intent.generative_models.CGenerativeModelSimulator import create_sim_params
from reaching_intent.generative_models.CGenerativeModelSimulator import scene_with_cabinet_and_two_objects
from reaching_intent.generative_models.CGenerativeModelSimulator import scene_with_table
from reaching_intent.generative_models.CGenerativeModelSimulator import scene_with_cabinet
from reaching_intent.generative_models.CGenerativeModelSimulator import scene_with_ur5table
###################################

###################################
# GENERIC PARAMETERS (tune for each application)
###################################
dataset_path = "datasets/dataset10K_2D_ur5_96p.dat"

# Desired number of data points that the dataset will contain
dataset_points = 1e4

# Max number of data points generated before saving to a file. Important to save batches when generating
# huge datasets to keep memory requirements bounded.
dataset_gen_batch = 1e3

# Read the generic parameters from the command line arguments
if len(sys.argv) > 3:
    dataset_path = sys.argv[1]
    dataset_points = int(sys.argv[2])
    dataset_gen_batch = sys.argv[3]
###################################

###################################
# APPLICATION SPECIFIC PARAMETERS (add/remove/edit parameters to fit your implementation)
###################################
sim_viz = False     # Visualize the generation process
sample_rate = 30    # Samples per second
sim_time = 3.2      # Duration of the simulated trajectories

# Generative model parameter limits: start volume (x,y,z), end volume (x,y,z), controller(Kp,Ki,Kd,iClamp,Krep)
param_limits_min = t_tensor([-0.06, 0.30, -0.10, 0.15, -0.3, 0.05, 15, 0.001, 0.01, 1.9, 19.0])
param_limits_max = t_tensor([-0.05, 0.31, -0.09, 0.80, 0.7, 0.06, 16, 0.002, 0.02, 2.0, 20.0])

# Select the parameters that are considered nuisance and the parameters that are considered interesting
latent_mask = t_tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]) == 1    # We are interested in the end position
nuisance_mask = t_tensor([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]) == 1  # The rest of the parameters are considered nuisance

# Get simulator parameters from the app specific import
gen_model_params = create_sim_params(sim_viz=sim_viz,
                                     sample_rate=sample_rate,
                                     sim_time=sim_time,
                                     model_path="pybullet_models/human_torso/model.urdf")

# Set up the scene
# scene_with_cabinet_and_two_objects(gen_model_params)
# scene_with_cabinet(gen_model_params)
scene_with_ur5table(gen_model_params)
###################################


#################################################
# GENERIC CODE (dataset generation)
# Some application specific code is included and properly commented
#################################################

# Uniform sampler to sample parameter values used by the generative model to generate data.
param_sampler = CSamplerUniform({"min": param_limits_min, "max": param_limits_max})

# Create an instance of a generative model used to generate synthetic data
gen_model = CGenerativeModelSimulator(gen_model_params)

# Check if the dataset is completely generated. Generate it otherwise
print("Load/generate dataset...")
dataset = CReachingDataset(filename=dataset_path, dataset_sample_rate=sample_rate, output_sample_rate=sample_rate)
dataset_size = len(dataset)


while dataset_size < dataset_points:
    print("Loaded dataset has %d data points. Generating more..." % dataset_size)
    dataset.samples.clear()

    for i in range(int(dataset_gen_batch)):
        t_ini = time.time()
        params = param_sampler.sample(nsamples=1, params=None)
        z = params[:, latent_mask]
        n = params[:, nuisance_mask]
        generated = gen_model.generate(z, n)

        # TODO: Solve this w/o application specific code
        # THIS IF IS APPLICATION SPECIFIC. CHECKS FOR THE VALIDITY OF THE DATA GENERATED
        # THE FUNCTIONALITY CAN BE EMBEDDED INTO THE GENERATIVE MODEL LOGIC THROUGH ITS CUSTOM PARAMS
        # THAT COULD HAVE A GENERIC SANITY CHECK FOR THE GENERATED DATA
        # Discard trajectories that do not end close to the goal
        if torch.sqrt(((generated[0][-3:] - z)*(generated[0][-3:] - z)).sum()) > 0.10:
            print("invalid trajectory")
            i -= 1
            continue

        dataset.samples.append([params, generated.view(-1)])
        print("Traj %d / %d. Time %2.4f" % (len(dataset) + dataset_size, dataset_points, time.time() - t_ini))

    dataset.dataset_save(dataset_path)
    dataset_size = dataset_size + len(dataset)

print("Generated dataset with %d data points." % dataset_size)
