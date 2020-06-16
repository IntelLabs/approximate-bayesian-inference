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
sim_viz = False      # Visualize the generation process
sample_rate = 30    # Samples per second

# Generative model parameter limits: start volume (x,y,z), end volume (x,y,z), controller(Kp,Ki,Kd,Krep,iClamp)
param_limits_min = t_tensor([-0.05, 0.30, -0.10, 0.25, -0.4, 0.20, 5, 0.005, 0, 0.10, 20])
param_limits_max = t_tensor([-0.04, 0.31, -0.09, 0.90, 0.4, 0.21, 20, 0.010, 0, 0.11, 30])

# Select the parameters that are considered nuisance and the parameters that are considered interesting
latent_mask = t_tensor([0,0,0,1,1,1,0,0,0,0,0]) == 1    # We are interested in the end position
nuisance_mask = t_tensor([0,0,0,1,1,1,0,0,0,0,0]) == 0  # The rest of the parameters are considered nuisance
###################################


#################################################
# GENERIC CODE (dataset generation)
#################################################
# Sampler to sample parameter values to generate trajectories
param_sampler = CSamplerUniform({"min": param_limits_min, "max": param_limits_max})

# Get simulator parameters from the app specific import
simulator_params = create_sim_params(sim_viz=sim_viz,
                                     sample_rate=sample_rate,
                                     model_path="pybullet_models/human_torso/model.urdf")

# Simulator used to generate synthetic data
neSimulator = CGenerativeModelSimulator(simulator_params)

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
        generated = neSimulator.generate(z, n)

        # Discard trajectories that do not end close to the goal
        if torch.sqrt(((generated[0][-3:] - z)*(generated[0][-3:] - z)).sum()) > 0.3:
            print("invalid trajectory")
            continue

        dataset.samples.append([params, generated.view(-1)])
        print("Traj %d / %d. Time %2.4f" % (len(dataset), dataset_points, time.time() - t_ini))

    dataset.dataset_save(dataset_path)
    dataset_size = dataset_size + len(dataset)

print("Generated dataset with %d data points." % dataset_size)
