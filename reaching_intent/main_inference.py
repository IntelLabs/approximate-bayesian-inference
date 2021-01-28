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
from utils.draw import draw_trajectory_diff
from utils.draw import draw_point
from utils.pybullet_utils import set_eef_position

from inference.CInferenceMetropolisHastings import CInferenceMetropolisHastings
from inference.CInferenceGrid import CInferenceGrid


if __name__ == "__main__":
    #################################################################################
    # APPLICATION SPECIFIC PARAMETERS
    #################################################################################
    sim_viz = False  # Show visualization of the simulator
    n_dims = 3      # Point dimensionality
    #################################################################################
    #################################################################################

    #################################################################################
    # LATENT and NUISANCE SPACES
    #################################################################################
    print("Set latent and nuisance spaces.")
    # Generative model parameter limits: start volume (x,y,z), end volume (x,y,z), controller(Kp,Ki,Kd,Krep,iClamp)
    param_limits_min = t_tensor([-0.06, 0.30, -0.10, 0.25, -0.4, 0.20, 1, 0.0, 0, 5.0, 0])
    param_limits_max = t_tensor([-0.05, 0.31, -0.09, 0.90, 0.4, 0.21, 10, 1.0, 10, 30.0, 2])

    # Select the parameters that are considered nuisance and the parameters that are considered interesting
    latent_mask = t_tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]) == 1  # We are interested in the end position
    nuisance_mask = t_tensor([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]) == 1  # The rest of the params are considered nuisance

    # Latent space
    z_min = param_limits_min[latent_mask]
    z_max = param_limits_max[latent_mask]
    latent_space = ContinousSpace(len(z_min), None, z_min, z_max)

    # Nuisance space (Hand initial position + Arm controller gains)
    n_min = param_limits_min[nuisance_mask]
    n_max = param_limits_max[nuisance_mask]
    nuisance_space = ContinousSpace(len(n_min), None, n_min, n_max)

    # Discretize the slack terms to be used for inference
    num_slacks = 50
    inference_slacks = torch.arange(1E-6, 20.0, 20.0 / num_slacks).double()
    inference_slacks = torch.exp(inference_slacks) * 1E-6
    #################################################################################
    #################################################################################

    #################################################################################
    # GENERATIVE MODEL SIMULATOR
    #################################################################################
    print("Load generative model: Simulator")
    # General parameters
    simulator_params = create_sim_params(sim_viz=sim_viz)
    gen_model_sim = CGenerativeModelSimulator(simulator_params)
    gen_model = gen_model_sim
    #################################################################################
    #################################################################################

    #################################################################################
    # GENERATIVE MODEL NEURAL EMULATOR
    #################################################################################
    print("Load generative model: Neural Surrogate")
    nn_model_path = "pytorch_models/ne_fc3_10k_MSE_in11_out450.pt"
    gen_model_neural_emulator = CGenerativeModelNeuralEmulator(nn_model_path)
    #################################################################################
    #################################################################################

    #################################################################################
    # PRIOR and PROPOSAL DISTRIBUTIONS
    #################################################################################
    prior_distribution = CSamplerUniform({"min": z_min, "max": z_max})
    nuisance_sampler = CSamplerUniform({"min": n_min, "max": n_max})
    proposal_distribution = CSamplerMultivariateNormal({"mean": torch.zeros_like(z_min),
                                                        "std": t_tensor([0.001, 0.001, 1e-6])})
    #################################################################################
    #################################################################################

    #################################################################################
    # OBSERVATION MODEL
    #################################################################################
    # TODO: Get data from an independent dataset not used for training
    print("Prepare observation model")
    observer_params = copy.deepcopy(simulator_params)
    observer_params["sigma"] = 0.001  # Additional gaussian noise added to observed trajectories
    observer_params["dataset_path"] = "./datasets/dataset10K.dat"
    observer_params["min_points"] = 3
    observer_params["obs_dimensions"] = 3
    observer_params["num_trajs"] = 100
    obs_model = CObservationModelDataset(observer_params)
    #################################################################################
    #################################################################################

    #################################################################################
    # INFERENCE ALGORITHM (MCMC-MH)
    #################################################################################
    neInferenceMCMC = CInferenceMetropolisHastings()

    # Configure inference
    inference_params = dict()
    inference_params["nsamples"] = 200
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
    inference_params["resolution"] = 0.03
    #################################################################################
    #################################################################################

    #################################################################################
    # SELECTION of generative model and inference algorithm
    #################################################################################
    gen_model = gen_model_neural_emulator
    # neInference = neInferenceGrid
    neInference = neInferenceMCMC
    # gen_model = gen_model_sim
    #################################################################################
    #################################################################################

    #################################################################################
    # VISUALIZATION and RESULTS
    #################################################################################
    if sim_viz is not None:
        visualizer = gen_model_sim.sim_id
    else:
        visualizer = None
    inference_params["visualizer"] = visualizer

    with open("results_%s_%s.dat" % (gen_model.get_name(), neInference.get_name()), "w") as f:
        f.write("Error      Time       %Observed  Slack      t_sample t_gens   t_lprob  #Eval  #Samples\n")
    #################################################################################
    #################################################################################

    # Draw ground truth
    if visualizer is not None:
        draw_trajectory(obs_model.traj.view(-1, n_dims), color=[0, 1, 0], draw_points=False, physicsClientId=visualizer)
        draw_point(obs_model.get_ground_truth()[latent_mask], [0, 1, 0], size=0.05, width=5, physicsClientId=visualizer)

    #################################################################################
    # INFERENCE LOOP
    #################################################################################
    viz_items = []
    iteration = 0
    while obs_model.is_ready():
        # Obtain observation and initialize latent space and nuisance values from their priors
        o = obs_model.get_observation()
        set_eef_position(o[-n_dims:], gen_model_sim.model_id, gen_model_sim.eef_link, physicsClientId=visualizer)
        z = prior_distribution.sample(1, None)
        n = nuisance_sampler.sample(1, None)

        # Nuisance starting position set to the first observed point
        n[0, 0:3] = o[0:3]

        # Draw current observation (purple) and ground truth trajectory (red)
        if visualizer is not None:
            for viz_item in viz_items:
                pybullet.removeUserDebugItem(viz_item, physicsClientId=visualizer)
            viz_items.extend(
                draw_trajectory(o.view(-1, n_dims), draw_points=False, width=8.0, color=[1, 0, 1],
                                physicsClientId=visualizer)
            )

        print("Run inference with %d observed points." % (len(o) / n_dims))
        t_inference = time.time()
        samples, likelihoods, stats = neInference.inference(obs=o, proposal=z, nuisance=n,
                                                                    gen_model=gen_model,
                                                                    likelihood_f=likelihood_f,
                                                                    params=inference_params,
                                                                    slacks=inference_slacks)
        runtime = time.time() - t_inference
        print("Done. Obtained %d samples in %fs" % (len(samples), runtime))

        #################################################################################
        # Evaluation and stats
        #  1 - L2 norm of the MAP predicted z and the ground truth z
        #  2 - Percent of the observed action
        #  3 - Inference time
        #  4 - Number of evaluated particles
        #  5 - Number of accepted particles (For MCMC approaches)
        #  6 - Grid size (For quasi-MC approaches)
        #################################################################################
        # Compute the maximum a posteriori particle
        idx = torch.argmax(likelihoods)
        idx_slack = int(idx / len(samples))
        idx_part = int(idx % len(samples))

        MAP_z = samples[idx_part]
        MAP_slack = inference_slacks[idx_slack]
        diff = obs_model.get_ground_truth()[latent_mask] - MAP_z
        error = torch.sqrt(torch.sum(diff * diff))
        traj_percent = float(len(o)) / len(obs_model.get_ground_truth_trajectory())

        debug_text = " Error: %2.4f \n Time: %2.4f \n PercentObserved: %2.4f \n #Samples: %d \n Slack: %2.6f \n Num Evals: %d \n Num Gens: %d" % (error, runtime, traj_percent, stats["nsamples"], MAP_slack, stats["nevals"], stats["ngens"])
        print("============================================")
        print(debug_text)
        print("============================================")
        with open("results_%s_%s.dat" % (gen_model.get_name(), neInference.get_name()), "a") as f:
            f.write("%2.8f %2.8f %2.8f %2.8f %2.6f %2.6f %2.6f %d  %d\n" % (error, runtime, traj_percent, MAP_slack, stats["tsamples"], stats["tgens"], stats["tevals"], stats["nevals"], stats["nsamples"]))

        if visualizer is not None:
            # for s in samples:
            #     viz_items.extend(draw_point(s, [0, 0, 1], size=0.01, width=2, physicsClientId=visualizer, lifetime=1.0))
            #     draw_point(s, [0, 0, 1], size=0.01, width=2, physicsClientId=visualizer, lifetime=10.0)
            viz_items.extend(draw_point(MAP_z, [1, 0, 0], size=0.05, width=5, physicsClientId=visualizer))
            # Draw generated trajectory for the GT point
            gen_traj = gen_model.generate(MAP_z.view(1, -1), n.view(1, -1))
            viz_items.extend(draw_trajectory(gen_traj.view(-1, n_dims), [1, 0, 0], draw_points=False, width=5))
            viz_items.extend(draw_trajectory_diff(obs_model.traj.view(-1, n_dims), gen_traj.view(-1, n_dims), [0, 0, .8], width=1))

        time.sleep(0.1)

        iteration = iteration + 1
        #################################################################################
        #################################################################################
    #################################################################################
    #################################################################################
