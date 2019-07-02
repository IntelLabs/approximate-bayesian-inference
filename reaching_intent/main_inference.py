#!/usr/bin/python3

#################
# SYSTEM IMPORTS
#################
import os
import sys
import time
import copy


#######################################
# GENERIC IMPORTS (no need to edit)
#######################################
from common.common import *

## Samplers
from samplers.CSamplerUniform import CSamplerUniform
from samplers.CSamplerMultivariateNormal import CSamplerMultivariateNormal
from samplers.CSamplerMixtureModel import CSamplerMixtureModel

## Inference methods
from inference.CInferenceMetropolisHastings import CInferenceMetropolisHastings
from inference.CInferenceSequentialMonteCarlo import CInferenceSMC
from inference.CInferenceTreePyramid import CInferenceQuadtree
from inference.CInferenceGrid import CInferenceGrid
from inference.CInferenceABCReject import CInferenceABCReject
from inference.CInferenceABCSequentialMonteCarlo import CInferenceABCSMC

## Likelihood function selection
from neural_emulators.loss_functions import log_likelihood_slacks as likelihood_f

## Generative models
from neural_emulators.generative_models import CGenerativeModelNeuralEmulator


##############################
# APPLICATION SPECIFIC IMPORTS (import from your application specific generative simulator and observation model)
##############################
from reaching_intent.generative_models.generative_models import CGenerativeModelSimulator as CGenerativeSimulator
# from reaching_intent_estimation.observation_models import CObservationModelDataset as CObservationModel
from reaching_intent.observation_models import CObservationModelContinousDataset as CObservationModel

# from manipulator_planning_control import pybullet_utils as pb


#######################################
# DEBUG DRAWING IMPORTS
#######################################
from mss import mss     # To take screenshots
import pyViewer         # Simple visualization engine
from utils.draw import draw_point
from utils.draw import draw_trajectory
from utils.draw import draw_text
###################################


############################
# GENERIC DEBUG PARAMETERS
############################
take_screenshot = True


###################################
# GENERIC PARAMETERS (tune for each application)
###################################
# Generative model selection. Options are: "sim", "emu"
generative_model = "emu"

# Neural emulator model path
# nn_model_path    = "models/test10k_gpu_MSE.pt"
nn_model_path    = "models/continous_table10K.pt"

# Select the inference methods to be used. Options are: ["grid", "quadtree", "smc", "abc-rejection", "abc-smc", "mcmc"]
inference_methods = []


# Select the grid size in m for each inference method. Required for grid and quadtree
inference_grid_sizes = []


# Discretize the slack terms to be used for inference
num_slacks = 50
inference_slacks = torch.arange(1E-9, 20.0, 20.0/num_slacks ).double()
inference_slacks = torch.exp(inference_slacks) * 1E-9


# Inference configuration parameters
MCMCnsamples   = 100        # Minimum number of accepted samples for MCMC inference
MCMCburn_in_samples = 50    # Number of samples discarded at the beginning of each MCMChain
nsamples = 50               # Number fo samples for pf and ABC
proposal_sigma = 0.01       # Sigma for the proposal distribution (used by MCMC, pf, ABC)
num_iters = 100             # Number of times inference will be performed for each selected method


# Read inference method and visualization flag from command line
if len(sys.argv) > 1:
    obs_viz = bool(int(sys.argv[1]))
    inference_methods.append(sys.argv[2])
else:
    obs_viz = True
    # inference_methods = ["mcmc", "grid", "smc", "abc-rejection", "abc-smc","quadtree"]
    inference_methods = ["quadtree"]
    inference_grid_sizes = [0.01, 0.03, 0.03, 0.03, 0.03, 0.03]


#########################################################################################################
# APPLICATION SPECIFIC PARAMETERS (add/remove/edit parameters to fit your implementation)
#########################################################################################################
model_path      = "models/human_torso/model.urdf"  # Path to the human torso model
sim_viz         = False                            # Show visualization of the simulator
sim_timestep    = 0.01                             # Simulator timestep
sample_rate     = 30
sim_time        = 3  # Prediction time window in seconds
n_prefix_points = 4  # Number of points in the prefix trajectory used for the generative model
n_obs_points    = 10  # Number of points used for observation. Includes prefix points
n_dims          = 3  # Point dimensionality
n_points        = sample_rate * sim_time

goal_min = t_tensor([0.25, -0.4, 0.2])
goal_max = t_tensor([0.9,   0.4, 0.21])
prefix   = torch.zeros(n_prefix_points * n_dims).double()
param_limits_min = torch.cat((prefix, goal_min))
param_limits_max = torch.cat((prefix, goal_max))

# param_limits_min = t_tensor([-0.1, 0.2, -0.2,    0.25, -0.4, 0.2,     10, 0, 0.001, 0.5])   # Goal volume lower bound
# param_limits_max = t_tensor([0.05, 0.5, 0.2,     0.9,   0.4, 0.21,    10.001, 0.001, 0.0011, 0.50001])   # Goal volume upper bound

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
simulator_params["robot_controller"] = None

# Uniform prior distribution
prior_sampler = CSamplerUniform({"min": param_limits_min, "max": param_limits_max})

# Gaussian proposal distribution
param_sampler_mean = torch.zeros_like(param_limits_min)
param_sampler_sigma = torch.ones_like(param_limits_min) * proposal_sigma

obs_sigma = 0.001               # Additional gaussian noise added to observed trajectories
device = torch.device("cpu")    # Force the computations on a specific device
###################################


#############################################################
# GENERIC CODE: Load simulator and emulator
#############################################################
neSimulator = CGenerativeSimulator(simulator_params)    # Simulator used to generate synthetic data

# Load neural likelihood surrogate
neNEmulator = CGenerativeModelNeuralEmulator(nn_model_path)
neNEmulator.model.move_to_device(device)

# Select generative method (simulator, neural-emulator)
if generative_model == "sim":
    generativeModel = neSimulator
elif generative_model == "emu":
    generativeModel = neNEmulator
else:
    raise ValueError("generative model: " + generative_model + " not supported.")


#############################################################
# GENERIC CODE: Load inference algorithms
#############################################################
# Instance inference algorithms
neInferenceGrid = CInferenceGrid(generativeModel, likelihood_f)
neInferenceQuad = CInferenceQuadtree(generativeModel, likelihood_f)
neInferenceSMC = CInferenceSMC(generativeModel, likelihood_f)
neInferenceABCReject = CInferenceABCReject(generativeModel, likelihood_f)
neInferenceABCSMC = CInferenceABCSMC(generativeModel, likelihood_f)
neInferenceMCMC = CInferenceMetropolisHastings(generativeModel, likelihood_f)

##################################################
# SPECIFIC CODE FOR RESULT VISUALIZATION AND DEBUG
##################################################
traj_ini = np.random.randint(0,9500)
traj_indices = range(traj_ini, traj_ini + num_iters)
# traj_indices = [5138]

# Parameters for the dataset based observation model
# real_trajectories_dataset = "../../hand_tracking/trajs.dat"
# simulator_trajetories_dataset = "./datasets/test.dat"
observer_params = copy.deepcopy(simulator_params)
observer_params["sigma"] = obs_sigma
observer_params["dataset_path"] = "./datasets/continous_table10K.dat"
observer_params["min_points"] = 3
observer_params["obs_dimensions"] = 3
observer_params["obs_window"] = 15
observer_params["goal_window"] = 120
neObserver = CObservationModel(observer_params)
if obs_viz:
    scene = pyViewer.CScene()
    scene.camera.alpha = 0.7
    scene.camera.beta = 0.7
    scene.camera.r = 4.0
    scene.camera.camera_matrix = scene.camera.look_at((0,0,0))


#####################################################################################
# Perform inference with selected: methods, generative model and observation model
#####################################################################################
for method, grid_size in zip(inference_methods, inference_grid_sizes):

    for iter in range(num_iters):
        experiment_prefix = method + str(grid_size) + generative_model
        print("Method: ", method, " grid size: ", grid_size)
        print("Slacks: ", inference_slacks)
        print ("Iteration %d / %d" % (iter, num_iters-1))
        print ("Gt traj index: %d" % (traj_indices[iter]))
        result_filename = "results/%03d_results_" % iter + experiment_prefix +".dat"
        f = open(result_filename, 'w')

        neObserver.new_trajectory(traj_indices[iter])

        obs_traj = []
        while len(obs_traj) < n_dims*n_obs_points:
            obs_traj = neObserver.get_observation()

        # The trajectory prefix forms the input for the generative model. Fix it to the observed prefix
        param_limits_min[0:n_dims * n_prefix_points] = obs_traj[0: n_dims * n_prefix_points]
        param_limits_max[0:n_dims * n_prefix_points] = obs_traj[0: n_dims * n_prefix_points]
        inference_proposal_sampler = CBoundedNormalSampler(param_limits_min, param_limits_max, param_sampler_mean, param_sampler_sigma)

        if obs_viz:
            scene.delete_graph(scene.root)
            pybullet_nodes = pyViewer.make_pybullet_scene(scene.ctx, physicsClientId=neSimulator.sim_id)
            scene.insert_graph(pybullet_nodes)
            # Example reference frame size 0.2
            nodes1 = pyViewer.CNode(geometry=pyViewer.make_mesh(scene.ctx, pyViewer.models.REFERENCE_FRAME_MESH, scale=0.2))
            scene.insert_graph([nodes1])

            # Example floor
            floor_node = pyViewer.CNode(geometry=pyViewer.make_mesh(scene.ctx, pyViewer.models.FLOOR_MESH, scale=1.0),
                                        transform=pyViewer.CTransform(pyViewer.tf.compose_matrix(translate=[0, 0, -0.65])))
            scene.insert_graph([floor_node])

        # time.sleep(1)

        iteration = 0
        error_hist = []
        runtime_hist = []
        traj_percent_hist = []
        grid_size_hist = []

        # Inference initialization for each loop if using particle filter and ABC-Reject
        neInferenceSMC.initialize(nsamples, prior_sampler)
        neInferenceABCReject.initialize(nsamples, prior_sampler)
        neInferenceABCSMC.initialize(nsamples, prior_sampler)
        neInferenceGrid.particles = None

        while neObserver.is_ready():

            obs_traj = neObserver.get_observation()

            # Update the trajectory prefix as the sliding time window moves forward
            param_limits_min[0:n_dims * n_prefix_points] = obs_traj[0 : n_dims * n_prefix_points]
            param_limits_max[0:n_dims * n_prefix_points] = obs_traj[0 : n_dims * n_prefix_points]
            inference_proposal_sampler = CBoundedNormalSampler(param_limits_min, param_limits_max, param_sampler_mean, param_sampler_sigma)

            if obs_viz:
                # Clear all debug items
                scene.clear()

                # Show GT goal
                draw_point(neObserver.get_goal(), [0, 1, 0, 1], size=0.05, width=5, physicsClientId=scene)

                # Show Observed trajectory
                draw_trajectory(obs_traj[n_dims * n_prefix_points:n_dims * n_obs_points].view(-1, n_dims),
                                [1, 0, 1, 1], 8, physicsClientId=scene, draw_points=False)

                # Show Prefix trajectory
                draw_trajectory(obs_traj[0:n_dims * n_prefix_points].view(-1, n_dims),
                                [0, 1, 0, 1], 8, physicsClientId=scene, draw_points=False)

                # Show GT trajectory
                # draw_trajectory(obs_traj.view(-1, n_dims),
                #                 [0, 1, 1, 1], 3, physicsClientId=scene, draw_points=False)


            t_inf = time.time()

            # Grid Inference
            if obs_viz:
                viewer = scene
            else:
                viewer = None

            #######################################################
            # RUN INFERENCE WITH THE DESIRED METHOD
            #######################################################
            # Grid Inference
            if method == "grid":
                neInferenceGrid.initialize(grid_size,param_limits_min,param_limits_max)
                samples, slack, nevalparts = neInferenceGrid.inference(obs=obs_traj.to(device), device=device,
                                                                       dim_min=param_limits_min, dim_max=param_limits_max,
                                                                       resolution=grid_size, slacks=inference_slacks, visualizer=viewer)
            # Quadtree Inference
            elif method == "quadtree":
                samples, slack, nevalparts = neInferenceQuad.inference(obs=obs_traj.to(device), device=device,
                                                                       dim_min=param_limits_min, dim_max=param_limits_max,
                                                                       resolution=grid_size, slacks=inference_slacks,
                                                                       EXPAND_THRESHOLD=-1, viewer=viewer)
            # Particle Filter Inference
            elif method == "smc":
                samples, slack, nevalparts = neInferenceSMC.inference(obs=obs_traj.to(device), nsamples=nsamples,
                                                                      device=device, visualizer=viewer,
                                                                      prior_sampler=prior_sampler, alpha=0.0,
                                                                      proposal_sampler=inference_proposal_sampler, slacks=inference_slacks)
            # MCMC Inference
            elif method == "mcmc":
                samples, slack, nevalparts = neInferenceMCMC.inference(obs=obs_traj.to(device), nsamples=MCMCnsamples,
                                                                       device=device, visualizer=viewer,
                                                                       prior_sampler=prior_sampler, proposal_sampler=inference_proposal_sampler,
                                                                       burn_in_samples=MCMCburn_in_samples)

            # Approximate Bayesian Computation - Rejection Inference
            elif method == "abc-rejection":
                samples, slack, nevalparts = neInferenceABCReject.inference(obs=obs_traj.to(device), nsamples=nsamples,
                                                                            device=device, visualizer=viewer,
                                                                            prior_sampler=prior_sampler, alpha=t_tensor([-1E+6]),
                                                                            proposal_sampler=inference_proposal_sampler)
            # Approximate Bayesian Computation- SMC Inference
            elif method == "abc-smc":
                samples, slack, nevalparts = neInferenceABCSMC.inference(obs=obs_traj.to(device), nsamples=nsamples,
                                                                         device=device, visualizer=viewer,
                                                                         prior_sampler=prior_sampler, alpha=t_tensor([-1E+7, -1E+6, -1E+5]),
                                                                         proposal_sampler=inference_proposal_sampler)
            else:
                raise Exception("Inference method not supported: " + method)

            params = samples.to(device)
            runtime = time.time() - t_inf
            #######################################################
            #######################################################


            # Evaluate error and  store statistics
            diff = neObserver.get_goal().to(params.device) - params[-n_dims:].double()
            error = torch.sqrt(torch.sum(diff*diff))
            traj_percent = float(len(obs_traj)) / len(neObserver.traj)
            error_hist.append(error)

            runtime_hist.append(runtime)
            traj_percent_hist.append(traj_percent)
            grid_size_hist.append(grid_size)
            debug_text = " Error: %2.4f \n Time: %2.4f \n PercentObserved: %2.4f \n GridSize: %2.4f \n Slack: %2.6f \n Num Evals: %d" % (error, runtime, traj_percent, grid_size, slack, nevalparts)
            print(debug_text)
            f.write("%2.8f %2.8f %2.8f %2.8f %2.8f %d\n" % (error, runtime, traj_percent, grid_size, slack, nevalparts))
            iteration = iteration + 1

            # Debug visualization
            if obs_viz:
                # Show inferred goals (Simulator generator)
                draw_point(params[-n_dims:], color=[1, 0, 0, 1], size=0.05, width=5, physicsClientId=viewer)

                # Show inferred trajectory
                nn_traj = generativeModel.generate(params.view(1, generativeModel.input_dims))[0]
                draw_trajectory(nn_traj[0:int(len(nn_traj) / 2)].view(-1, n_dims).detach(), [1, 0, 0], width=3,
                                physicsClientId=viewer, draw_points=False)

                # Take a screenshot
                if take_screenshot:
                    with mss() as sct:
                        video_output_dir = "inference_video/" + experiment_prefix + "_%fm/iteration_%d/" % (
                        grid_size, iter)
                        if not os.path.exists(video_output_dir):
                            os.makedirs(video_output_dir)
                        sct.shot(mon=2, output=video_output_dir + "frame_%d.png" % iteration)

                for event in scene.get_events():
                    if event.type == pyViewer.CEvent.QUIT:
                        quit()

                    scene.process_event(event)
                pyViewer.update_pybullet_nodes(pybullet_nodes, physicsClientId=neSimulator.sim_id)

                # Show debug text
                draw_text(text=debug_text, position=(0,20*6,1), visualizer=viewer, color=(1,1,0))

                scene.draw()

        f.close()
