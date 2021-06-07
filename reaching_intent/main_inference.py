#!/usr/bin/python3
import copy
import time
import pybullet
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from common.common import *
from reaching_intent.generative_models.CGenerativeModelSimulator import CGenerativeModelSimulator
from reaching_intent.generative_models.CGenerativeModelSimulator import create_sim_params
from reaching_intent.generative_models.CGenerativeModelSimulator import scene_with_cabinet_and_two_objects
from reaching_intent.generative_models.CGenerativeModelSimulator import scene_with_table
from reaching_intent.generative_models.CGenerativeModelSimulator import scene_with_cabinet
from reaching_intent.generative_models.CGenerativeModelSimulator import scene_with_ur5table

from neural_emulators.CGenerativeModelNeuralEmulator import CGenerativeModelNeuralEmulator
from samplers.CSamplerUniform import CSamplerUniform
from reaching_intent.observation_models.CObservationModelDataset import CObservationModelDataset
from utils.draw import draw_trajectory
from utils.draw import draw_trajectory_diff
from utils.draw import draw_point
from utils.pybullet_utils import set_eef_position

from inference.CInferenceGrid import CInferenceGrid

from ais_benchmarks.sampling_methods import CTreePyramidSampling
from ais_benchmarks.sampling_methods import CMetropolisHastings
from ais_benchmarks.distributions import ABCDistribution
from ais_benchmarks.distributions import CMultivariateDelta
from ais_benchmarks.distributions import CMultivariateNormal
from ais_benchmarks.distributions import CMultivariateUniform
from ais_benchmarks.distributions import CGaussianMixtureModel
from ais_benchmarks.distributions import CDistribution
from ais_benchmarks.distributions import GenericNuisanceGenModel

from scipy.spatial import distance


def loglikelihood_f(x, trajs, slack=t_tensor([0.01])):
    k = len(x)  # Observation dimensions

    l = len(trajs[0])  # Emulator dimensions

    if k > l:
        k = l
        x = x[0:k]

    slack = slack.reshape(-1, 1)
    term1 = np.array([-(k / 2) * math.log(2 * np.pi)])
    diff = distance.cdist(x.reshape((1, -1)), trajs[:, 0:k], 'sqeuclidean')
    term2 = k * np.log(1 / (slack * slack)) / 2
    term3 = - diff / (slack * slack) / 2
    loss = term1 + term2 + term3
    return loss


if __name__ == "__main__":
    #################################################################################
    # APPLICATION SPECIFIC PARAMETERS
    #################################################################################
    sim_viz = False  # Show visualization of the simulator
    n_dims = 3  # Point dimensionality
    make_plot = True  # Create a 3D plot of each inference step
    n_samples = 100
    sample_rate = 30
    traj_duration = 3.2
    #################################################################################
    #################################################################################

    #################################################################################
    # LATENT and NUISANCE SPACES
    #################################################################################
    print("Set latent and nuisance spaces.")
    # Generative model parameter limits: start volume (x,y,z), end volume (x,y,z), controller(Kp,Ki,Kd,Krep,iClamp)
    param_limits_min = t_tensor([-0.06, 0.30, -0.10, 0.15, -0.3, 0.05, 1, 0.0, 0, 0.10, 0])
    param_limits_max = t_tensor([-0.05, 0.31, -0.09, 0.80, 0.7, 0.06, 10, 1.0, 10, 0.35, 2])

    # Select the parameters that are considered nuisance and the parameters that are considered interesting
    latent_mask = t_tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]) == 1  # We are interested in the end position
    nuisance_mask = t_tensor([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]) == 1  # The rest of the params are considered nuisance

    # Latent space
    z_min = param_limits_min[latent_mask]
    z_max = param_limits_max[latent_mask]

    # Nuisance space (Hand initial position + Arm controller gains)
    n_min = param_limits_min[nuisance_mask]
    n_max = param_limits_max[nuisance_mask]
    #################################################################################
    #################################################################################

    #################################################################################
    # SLACK GRID EVALUATION POINTS
    #################################################################################
    # num_slacks = 20
    # inference_slacks = torch.arange(1E-2, 20.0, 20.0 / num_slacks).double()
    # inference_slacks = torch.exp(inference_slacks) * 1E-2
    inference_slacks = np.array([[0.3]])
    #################################################################################
    #################################################################################

    #################################################################################
    # GENERATIVE MODEL SIMULATOR (also used for visualization)
    #################################################################################
    print("Load generative model: Simulator")
    simulator_params = create_sim_params(sim_viz=sim_viz, sample_rate=sample_rate, sim_time=traj_duration)
    scene_with_ur5table(simulator_params)
    gen_model_sim = CGenerativeModelSimulator(simulator_params)
    #################################################################################
    #################################################################################

    #################################################################################
    # GENERATIVE MODEL NEURAL EMULATOR
    #################################################################################
    print("Load generative model: Neural Surrogate")
    nn_model_path = "pytorch_models/ne_fc4_10k2D_MSE_in11_out288.pt"
    gen_model_neural_emulator = CGenerativeModelNeuralEmulator(nn_model_path)
    #################################################################################
    #################################################################################

    #################################################################################
    # PRIOR DISTRIBUTIONS
    #################################################################################
    prior_distribution = CSamplerUniform({"min": z_min, "max": z_max})
    nuisance_sampler = CSamplerUniform({"min": n_min, "max": n_max})
    #################################################################################
    #################################################################################

    #################################################################################
    # OBSERVATION MODEL
    #################################################################################
    # TODO: Get data from an independent dataset not used for training
    print("Prepare observation model")
    observer_params = copy.deepcopy(simulator_params)
    observer_params["sigma"] = 0.001  # Additional gaussian noise added to observed trajectories
    observer_params["dataset_path"] = "./datasets/dataset10K_2D_ur5_96p.dat"
    observer_params["min_points"] = 3
    observer_params["obs_dimensions"] = 3
    observer_params["num_trajs"] = 100
    obs_model = CObservationModelDataset(observer_params)
    # obs_model.new_trajectory(12)
    #################################################################################
    #################################################################################

    #################################################################################
    # INFERENCE ALGORITHM (Grid)
    #################################################################################
    neInferenceGrid = CInferenceGrid()
    neInferenceGrid.name = "grid"
    inference_params = dict()
    inference_params["z_min"] = z_min
    inference_params["z_max"] = z_max
    inference_params["resolution"] = 0.01
    #################################################################################
    #################################################################################

    #################################################################################
    # INFERENCE ALGORITHM (TP-AIS)
    #################################################################################
    inference_params["space_min"] = z_min.numpy()
    inference_params["space_max"] = z_max.numpy()
    inference_params["ess_target"] = 0.8
    inference_params["n_min"] = 5
    inference_params["method"] = "simple"
    inference_params["resampling"] = "leaf"
    inference_params["kernel"] = "haar"
    inference_params["dims"] = len(z_min)
    inference_params["parallel_samples"] = 32
    neInferenceTPAIS = CTreePyramidSampling(inference_params)
    #################################################################################
    #################################################################################

    #################################################################################
    # INFERENCE ALGORITHM (MCMC - Metropolis Hastings)
    #################################################################################
    inference_params["space_min"] = z_min.numpy()
    inference_params["space_max"] = z_max.numpy()
    inference_params["n_samples_kde"] = 300
    inference_params["kde_bw"] = 0.01
    inference_params["proposal_sigma"] = 0.01
    inference_params["n_steps"] = 1
    inference_params["dims"] = len(z_min)
    inference_params["n_burnin"] = 10
    neInferenceMH = CMetropolisHastings(inference_params)
    #################################################################################
    #################################################################################

    #################################################################################
    # SELECTION of generative model and inference algorithm
    #################################################################################
    gen_model = gen_model_neural_emulator
    # gen_model = gen_model_sim
    neInference = neInferenceGrid
    # neInference = neInferenceTPAIS
    # neInference = neInferenceMH
    #################################################################################
    #################################################################################

    #################################################################################
    # WRAP GENERATIVE MODEL INTO A DISTRIBUTION
    #################################################################################
    gen_dist = GenericNuisanceGenModel(
        {"gen_function": gen_model.generate,
         "params_mask": latent_mask,
         "nuisance_dist": CMultivariateDelta({"center": ((param_limits_max + param_limits_min) / 2)[nuisance_mask],
                                              "support": [param_limits_min[nuisance_mask],
                                                          param_limits_max[nuisance_mask]]}),
         "dims": n_dims,
         "support": [z_min.numpy(), z_max.numpy()],
         "noise_sigma": 0.001}
    )
    #################################################################################
    #################################################################################

    #################################################################################
    # VISUALIZATION and RESULTS
    #################################################################################
    if sim_viz:
        visualizer = gen_model_sim.sim_id
    else:
        visualizer = None
    inference_params["visualizer"] = visualizer

    with open("results_%s_%s.dat" % (gen_model.get_name(), neInference.name), "w") as f:
        f.write("Error      Time       %Observed  Slack      t_sample t_gens   t_lprob  #Eval  #Samples\n")

    # Draw ground truth
    if visualizer is not None:
        draw_trajectory(obs_model.traj.view(-1, n_dims), color=[0, 1, 0], draw_points=False, physicsClientId=visualizer)
        draw_point(obs_model.get_ground_truth()[latent_mask], [0, 1, 0], size=0.05, width=5, physicsClientId=visualizer)
    #################################################################################
    #################################################################################

    # for i in range(25):
    #     # Obtain observation and initialize latent space and nuisance values from their priors
    #     o = obs_model.get_observation()

    #################################################################################
    # INFERENCE LOOP
    #################################################################################
    if make_plot is True:
        fig = plt.figure()
    viz_items = []
    iteration = 0
    # z = prior_distribution.sample(1, None)
    frame = None
    while obs_model.is_ready():
        # Obtain observation and initialize latent space and nuisance values from their priors
        o = obs_model.get_observation()
        set_eef_position(o[-n_dims:], gen_model_sim.model_id, gen_model_sim.eef_link, physicsClientId=gen_model_sim.sim_id)
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
        # target_d_params = dict()
        # target_d_params["dims"] = 3
        # target_d_params["likelihood_f"] = lambda x: np.exp(loglikelihood_f(x))
        # target_d_params["loglikelihood_f"] = loglikelihood_f
        # target_d_params["support"] = [z_min.numpy(), z_max.numpy()]
        # target_d_params["prior_d"] = CMultivariateUniform({"center": (z_max.numpy() + z_min.numpy()) / 2,
        #                                                    "radius": (z_max.numpy() - z_min.numpy()) / 2})
        # target_d_params["sensor_d"] = CMultivariateDelta({"center": o.numpy(),
        #                                                   "support": [o.numpy() - 1, o.numpy() + 1]})
        # target_d_params["gen_d"] = gen_dist
        # target_d_params["slack"] = np.array([0.1])
        # target_d = ABCDistribution(target_d_params)
        # target_d.condition(o.numpy())

        # neInference.reset()
        # samples, weights = neInference.importance_sample(target_d, n_samples, timeout=60.0)
        samples, weights, stats = neInference.inference(obs=o, nuisance=n,
                                                        gen_model=gen_model,
                                                        likelihood_f=loglikelihood_f,
                                                        params=inference_params,
                                                        slacks=inference_slacks)
        runtime = time.time() - t_inference
        print("Done. Obtained %d samples in %fs" % (len(samples), runtime))

        # AGGREGATED STATS FROM EACH FRAME
        stats = neInference.get_stats()
        stats["time"] = runtime
        if frame is None:
            frame = [list(stats.values())]
        else:
            frame.append(list(stats.values()))
        df = pd.DataFrame(data=frame, columns=stats.keys())
        df_stats = df.mean().append(df.std(), )
        print(df_stats)

        # Compute the maximum a posteriori particle, without considering multiple slacks
        idx = np.argmax(weights)
        MAP_z = samples[idx]
        diff = obs_model.get_ground_truth()[latent_mask] - MAP_z
        error = torch.sqrt(torch.sum(diff * diff))
        traj_percent = float(len(o)) / len(obs_model.get_ground_truth_trajectory())

        # TODO: REACTIVATE SAMPLING STATS OF ACTIVE FRAME
        #################################################################################
        # Evaluation and stats
        #  1 - L2 norm of the MAP predicted z and the ground truth z
        #  2 - Percent of the observed action
        #  3 - Inference time
        #  4 - Number of evaluated particles
        #  5 - Number of accepted particles (For MCMC approaches)
        #  6 - Grid size (For quasi-MC approaches)
        #################################################################################

        # # Compute the maximum a posteriori particle
        # idx = torch.argmax(weights)
        # idx_slack = int(idx / len(samples))
        # idx_part = int(idx % len(samples))
        #
        # MAP_z = samples[idx_part]
        # MAP_slack = inference_slacks[idx_slack]
        # diff = obs_model.get_ground_truth()[latent_mask] - MAP_z
        # error = torch.sqrt(torch.sum(diff * diff))
        # traj_percent = float(len(o)) / len(obs_model.get_ground_truth_trajectory())
        # z = MAP_z.view(1, -1)

        # debug_text = " Error: %2.4f \n Time: %2.4f \n PercentObserved: %2.4f \n #Samples: %d \n Slack: %2.6f \n Num Evals: %d \n Num Gens: %d" % \
        #              (error, runtime, traj_percent, stats["nsamples"], MAP_slack, stats["nevals"], stats["ngens"])
        # print("============================================")
        # print(debug_text)
        # print("============================================")
        # with open("results/results_%s_%s.dat" % (gen_model.get_name(), neInference.get_name()), "a") as f:
        #     f.write("%2.8f %2.8f %2.8f %2.8f %2.6f %2.6f %2.6f %d  %d\n" % (error, runtime, traj_percent, MAP_slack,
        #                                                                     stats["tsamples"], stats["tgens"],
        #                                                                     stats["tevals"], stats["nevals"],
        #                                                                     stats["nsamples"]))

        if visualizer is not None:
            for s in samples:
                viz_items.extend(draw_point(s, [0, 0, 1], size=0.01, width=2, physicsClientId=visualizer))
                # draw_point(s, [0, 0, 1], size=0.01, width=2, physicsClientId=visualizer, lifetime=10.0)
            viz_items.extend(draw_point(MAP_z, [1, 0, 0], size=0.05, width=5, physicsClientId=visualizer))
            # Draw generated trajectory for the GT point
            gen_traj = gen_model.generate(MAP_z.reshape(1, -1), n.view(1, -1))
            viz_items.extend(draw_trajectory(gen_traj.view(-1, n_dims), [1, 0, 0], draw_points=False, width=5))
            viz_items.extend(draw_trajectory_diff(obs_model.traj.view(-1, n_dims), gen_traj.view(-1, n_dims),
                                                  [0, 0, .8], width=1))

        if make_plot is True:
            plt.clf()
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=30, azim=160)
            ax.set_xlim(-.2, 1)
            ax.set_ylim(-.7, .7)
            ax.set_zlim(0, 1)
            viz_indices = weights > -100
            samples_viz = samples[viz_indices]
            colors = cm.hot(np.exp((weights - weights.max())))
            alphas = np.exp((weights - weights.max()))
            colors[:, 3] = np.clip(alphas, 0.05, 1.0)
            ax.scatter(xs=samples_viz[:, 0], ys=samples_viz[:, 1], zs=samples_viz[:, 2], c=colors[viz_indices])
            ax.plot(xs=o.view(-1, n_dims)[:, 0], ys=o.view(-1, n_dims)[:, 1], zs=o.view(-1, n_dims)[:, 2], c='purple')

            gen_traj = gen_model.generate(MAP_z.reshape(1, -1), n.view(1, -1)).cpu().detach()
            ax.plot(xs=gen_traj.view(-1, n_dims)[:, 0], ys=gen_traj.view(-1, n_dims)[:, 1], zs=gen_traj.view(-1, n_dims)[:, 2], c='red')
            ax.plot(xs=[0, .1], ys=[0, 0], zs=[0, 0], c='r')
            ax.plot(xs=[0, 0], ys=[0, .1], zs=[0, 0], c='g')
            ax.plot(xs=[0, 0], ys=[0, 0], zs=[0, .1], c='b')
            neInference.draw(ax)

            gt_point = obs_model.get_ground_truth()[latent_mask]
            ax.scatter(xs=[gt_point[0]], ys=[gt_point[1]], zs=[gt_point[2]], c='b', marker='*')
            # ax.scatter(xs=[MAP_z[0]], ys=[MAP_z[1]], zs=[MAP_z[2]], c='k', marker='*')
            ax.plot(xs=[gt_point[0], MAP_z[0]], ys=[gt_point[1], MAP_z[1]], zs=[gt_point[2], MAP_z[2]], c='b',
                    linestyle="--", alpha=0.5)

            # debug_text = "Err: %2.4f T: %2.4f Obs%%: %2.4f #Samples: %d AccRate: %2.4f $\epsilon$: %2.6f NEvals: %d NGens: %d" % \
            #              (
            #                  error, runtime, traj_percent, stats["nsamples"], stats["nsamples"] / stats["ngens"],
            #                  MAP_slack,
            #                  stats["nevals"], stats["ngens"])
            debug_text = "Err: %2.4f T: %2.4f Obs%%: %2.4f #Samples: %d" % (error, runtime, traj_percent, len(samples))
            plt.title(debug_text, fontsize=8)
            plt.show(block=False)
            plt.pause(0.01)
            plt.savefig("results/inference_%s_it%d.png" % (neInference.name, iteration), dps=700)
        else:
            time.sleep(0.01)

        iteration = iteration + 1
        #################################################################################
        #################################################################################
    #################################################################################
    #################################################################################
