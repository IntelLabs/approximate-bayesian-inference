#!/usr/bin/python3
import copy
import time
import pybullet

from common.common import *
from reaching_intent.generative_models.CGenerativeModelSimulator import CGenerativeModelSimulator
from neural_emulators.CGenerativeModelNeuralEmulator import CGenerativeModelNeuralEmulator
from samplers.CSamplerUniform import CSamplerUniform
from samplers.CSamplerMultivariateNormal import CSamplerMultivariateNormal
from reaching_intent.observation_models.CObservationModelDataset import CObservationModelDataset
from inference.CInferenceMetropolisHastings import CInferenceMetropolisHastings
from neural_emulators.loss_functions import log_likelihood_slacks as likelihood_f
from spaces.ContinousSpace import ContinousSpace
from utils import pybullet_controller as pbc
from utils import pybullet_utils as pb
from utils.draw import draw_trajectory


if __name__ == "__main__":
    #################################################################################
    # APPLICATION SPECIFIC PARAMETERS
    #################################################################################
    models_path = "/home/jfelip/workspace/prob-comp-code/approximate_bayesian_inference/reaching_intent/pybullet_models/"
    model_path = "human_torso/model.urdf"           # Path to the human torso model
    sim_viz = True                                 # Show visualization of the simulator
    sim_timestep = 0.01                             # Simulator timestep
    sample_rate = 30                                # Trajectory point sample rate
    sim_time = 3                                    # Prediction time window in seconds
    n_prefix_points = 4                             # Number of points in the prefix trajectory used for the gen model
    n_obs_points = 10                               # Number of points used for observation. Includes prefix points
    n_points = sample_rate * sim_time               # Total number of points in a trajectory
    n_dims = 3                                      # Point dimensionality
    #################################################################################
    #################################################################################

    #################################################################################
    # LATENT and NUISANCE SPACES
    #################################################################################
    # Latent space
    z_min = t_tensor([0.25, -0.4, 0.2])
    z_max = t_tensor([0.9, 0.4, 0.21])
    latent_space = ContinousSpace(len(z_min), None, z_min, z_max)

    # Nuisance space (Hand initial position + Arm controller gains)
    ini_x = 0.05
    ini_y = 0.3
    ini_z = 0.0
    Kp = 5          # Proportional gain
    Ki = 0.01       # Integral gain
    Kd = 0          # Derivative gain
    iClamp = 25.0   # Maximum contribution of the integral term
    Krep = 0.05      # Obstacle repulsive gain
    n_min = t_tensor([ini_x, ini_y, ini_z, Kp, Ki, Kd, Krep, iClamp])
    n_max = t_tensor([ini_x, ini_y, ini_z, Kp, Ki, Kd, Krep, iClamp])
    nuisance_space = ContinousSpace(len(n_min), None, n_min, n_max)

    # Join latent + nuisance
    param_limits_min = torch.cat((n_min, z_min))
    param_limits_max = torch.cat((n_max, z_max))
    #################################################################################
    #################################################################################

    #################################################################################
    # GENERATIVE MODEL SIMULATOR
    #################################################################################
    # General parameters
    simulator_params = dict()
    simulator_params["robot_model_path"] = models_path + model_path
    simulator_params["visualization"] = sim_viz
    simulator_params["timestep"] = sim_timestep
    simulator_params["episode_time"] = sim_time
    simulator_params["sample_rate"] = sample_rate
    simulator_params["sim_id"] = 0

    # Obstacles
    simulator_objects = dict()
    simulator_objects["path"] = [models_path + "table/table.urdf"]
    simulator_objects["pose"] = [[0.6, 0, -0.65]]
    simulator_objects["static"] = [True]
    simulator_params["objects"] = simulator_objects

    # Arm controller (PID + ForceField obstacle avoidance)
    PIDcontroller = pbc.CPIDController(Kp=Kp, Ki=Ki, Kd=Kd, iClamp=iClamp)
    controller = pbc.CPotentialFieldController(Krep=Krep, ctrl=PIDcontroller)
    simulator_params["robot_controller"] = controller
    gen_model_sim = CGenerativeModelSimulator(simulator_params)

    # Set the resting position for each arm joint
    joint_indices = pb.get_actuable_joint_indices(gen_model_sim.model_id, gen_model_sim.sim_id)
    joint_rest_positions = torch.zeros(len(joint_indices))
    joint_rest_positions[5] = -1.04  # Elbow at 60 deg
    joint_rest_positions[6] = 1.57   # Palm down
    gen_model_sim.joint_rest_positions = joint_rest_positions
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
                                                        "std": t_tensor([0.002, 0.002, 1e-6])})
    #################################################################################
    #################################################################################

    #################################################################################
    # OBSERVATION MODEL
    #################################################################################
    obs_sigma = 0.001  # Additional gaussian noise added to observed trajectories
    observer_params = copy.deepcopy(simulator_params)
    observer_params["sigma"] = obs_sigma
    observer_params["dataset_path"] = "./datasets/default.dat"
    observer_params["min_points"] = 3
    observer_params["obs_dimensions"] = 3
    obs_model = CObservationModelDataset(observer_params)
    #################################################################################
    #################################################################################

    #################################################################################
    # INFERENCE ALGORITHM
    #################################################################################
    MCMCnsamples = 100
    MCMCburn_in_samples = 20
    neInferenceMCMC = CInferenceMetropolisHastings()
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

    gen_model = gen_model_neural_emulator
    # gen_model = gen_model_sim
    while obs_model.is_ready():
        # Obtain observation and initialize latent space and nuisance values from their priors
        o = obs_model.get_observation()
        z = prior_distribution.sample(1, None)
        n = nuisance_sampler.sample(1, None)

        # Configure inference algorithm
        inference_params = dict()
        inference_params["nsamples"] = MCMCnsamples
        inference_params["burn_in"] = MCMCburn_in_samples
        inference_params["proposal_dist"] = proposal_distribution

        # Nuisance starting position set to the first observed point
        n[0, 0:3] = o[0:3]

        # Draw current observation (purple) and ground truth trajectory (red)
        if visualizer is not None:
            pybullet.removeAllUserDebugItems(physicsClientId=visualizer)
            draw_trajectory(obs_model.traj.view(-1, n_dims), draw_points=False, physicsClientId=visualizer)
            draw_trajectory(o.view(-1, n_dims), draw_points=True, width=3.0, color=[1, 0, 1], physicsClientId=visualizer)

        print("Run inference with %d observed points." % (len(o) / n_dims))
        t_inference = time.time()
        samples, slack, nevalparts = neInferenceMCMC.inference(obs=o, proposal=z, nuisance=n,
                                                               gen_model=gen_model,
                                                               likelihood_f=likelihood_f,
                                                               params=inference_params,
                                                               visualizer=visualizer)
        runtime = time.time() - t_inference
        print("Done. Obtained %d samples in %fs" % (len(samples), runtime))
