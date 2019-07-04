#!/usr/bin/python3
import copy
import time
import pybullet

from common.common import *
from reaching_intent.generative_models.CGenerativeModelSimulator import CGenerativeModelSimulator
from reaching_intent.generative_models.CGenerativeModelSimulator import create_sim_params
from neural_emulators.CGenerativeModelNeuralEmulator import CGenerativeModelNeuralEmulator
from samplers.CSamplerUniform import CSamplerUniform
from samplers.CSamplerMultivariateNormal import CSamplerMultivariateNormal
from reaching_intent.observation_models.CObservationModelDataset import CObservationModelDataset
from neural_emulators.loss_functions import log_likelihood_slacks as likelihood_f
from spaces.ContinousSpace import ContinousSpace
from utils.draw import draw_trajectory

from inference.CInferenceMetropolisHastings import CInferenceMetropolisHastings
from inference.CInferenceGrid import CInferenceGrid


if __name__ == "__main__":
    #################################################################################
    # APPLICATION SPECIFIC PARAMETERS
    #################################################################################
    sim_viz = True  # Show visualization of the simulator
    n_dims = 3      # Point dimensionality
    #################################################################################
    #################################################################################

    #################################################################################
    # LATENT and NUISANCE SPACES
    #################################################################################
    # Generative model parameter limits: start volume (x,y,z), end volume (x,y,z), controller(Kp,Ki,Kd,Krep,iClamp)
    param_limits_min = t_tensor([-0.05, 0.30, -0.10, 0.25, -0.4, 0.20, 5, 0.005, 0, 0.10, 20])
    param_limits_max = t_tensor([-0.04, 0.31, -0.09, 0.90, 0.4, 0.21, 20, 0.010, 0, 0.11, 30])

    # Select the parameters that are considered nuisance and the parameters that are considered interesting
    latent_mask = t_tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]) == 1  # We are interested in the end position
    nuisance_mask = t_tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]) == 0  # The rest of the params are considered nuisance

    # Latent space
    z_min = param_limits_min[latent_mask]
    z_max = param_limits_max[latent_mask]
    latent_space = ContinousSpace(len(z_min), None, z_min, z_max)

    # Nuisance space (Hand initial position + Arm controller gains)
    n_min = param_limits_min[nuisance_mask]
    n_max = param_limits_max[nuisance_mask]
    nuisance_space = ContinousSpace(len(n_min), None, n_min, n_max)
    #################################################################################
    #################################################################################

    #################################################################################
    # GENERATIVE MODEL SIMULATOR
    #################################################################################
    # General parameters
    simulator_params = create_sim_params(sim_viz=sim_viz)
    gen_model_sim = CGenerativeModelSimulator(simulator_params)
    gen_model = gen_model_sim
    #################################################################################
    #################################################################################

    #################################################################################
    # GENERATIVE MODEL NEURAL EMULATOR
    #################################################################################
    nn_model_path = "pytorch_models/test10k_gpu_MSE_2.pt"
    gen_model_neural_emulator = CGenerativeModelNeuralEmulator(nn_model_path)
    #################################################################################
    #################################################################################

    #################################################################################
    # PRIOR and PROPOSAL DISTRIBUTIONS
    #################################################################################
    prior_distribution = CSamplerUniform({"min": z_min, "max": z_max})
    nuisance_sampler = CSamplerUniform({"min": n_min, "max": n_max})
    proposal_distribution = CSamplerMultivariateNormal({"mean": torch.zeros_like(z_min),
                                                        "std": t_tensor([0.01, 0.01, 1e-6])})
    #################################################################################
    #################################################################################

    #################################################################################
    # OBSERVATION MODEL
    #################################################################################
    observer_params = copy.deepcopy(simulator_params)
    observer_params["sigma"] = 0.001  # Additional gaussian noise added to observed trajectories
    observer_params["dataset_path"] = "./datasets/default.dat"
    observer_params["min_points"] = 3
    observer_params["obs_dimensions"] = 3
    obs_model = CObservationModelDataset(observer_params)
    #################################################################################
    #################################################################################

    #################################################################################
    # INFERENCE ALGORITHM (MCMC-MH)
    #################################################################################
    neInferenceMCMC = CInferenceMetropolisHastings()

    # Configure inference
    inference_params = dict()
    inference_params["nsamples"] = 100
    inference_params["burn_in"] = 20
    inference_params["proposal_dist"] = proposal_distribution
    inference_params["z_min"] = z_min
    inference_params["z_max"] = z_max
    inference_params["timeout"] = 10
    #################################################################################
    #################################################################################

    #################################################################################
    # INFERENCE ALGORITHM (Grid)
    #################################################################################
    neInferenceGrid = CInferenceGrid()
    inference_params["z_min"] = z_min
    inference_params["z_max"] = z_max
    inference_params["resolution"] = 0.04
    #################################################################################
    #################################################################################

    #################################################################################
    # VISUALIZATION
    #################################################################################
    if sim_viz is not None:
        visualizer = gen_model_sim.sim_id
    else:
        visualizer = None
    #################################################################################
    #################################################################################

    # Select generative model and inference method
    gen_model = gen_model_neural_emulator
    # neInference = neInferenceGrid
    neInference = neInferenceMCMC
    # gen_model = gen_model_sim
    while obs_model.is_ready():
        # Obtain observation and initialize latent space and nuisance values from their priors
        o = obs_model.get_observation()
        z = prior_distribution.sample(1, None)
        n = nuisance_sampler.sample(1, None)

        # Nuisance starting position set to the first observed point
        n[0, 0:3] = o[0:3]

        # Draw current observation (purple) and ground truth trajectory (red)
        if visualizer is not None:
            pybullet.removeAllUserDebugItems(physicsClientId=visualizer)
            draw_trajectory(obs_model.traj.view(-1, n_dims), draw_points=False, physicsClientId=visualizer)
            draw_trajectory(o.view(-1, n_dims), draw_points=True, width=3.0, color=[1, 0, 1], physicsClientId=visualizer)

        print("Run inference with %d observed points." % (len(o) / n_dims))
        t_inference = time.time()
        samples, slack, nevalparts = neInference.inference(obs=o, proposal=z, nuisance=n,
                                                           gen_model=gen_model,
                                                           likelihood_f=likelihood_f,
                                                           params=inference_params,
                                                           slacks=t_tensor([1E-4]),
                                                           visualizer=visualizer)
        runtime = time.time() - t_inference
        time.sleep(0.5)
        print("Done. Obtained %d samples in %fs" % (len(samples), runtime))
