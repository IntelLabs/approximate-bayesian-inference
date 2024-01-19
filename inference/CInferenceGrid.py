
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT

import time

from common.common import *
from common.CBaseInferenceAlgorithm import CBaseInferenceAlgorithm
from utils.draw import draw_samples


class CInferenceGrid(CBaseInferenceAlgorithm):
    def __init__(self):
        super(CInferenceGrid, self).__init__()
        self.stats = dict()
        self.trajs = None
        self.particles = None

    @staticmethod
    def get_name():
        return "grid"

    @staticmethod
    def grid_latent_variables(dim_min, dim_max, resolution):
        dim_range = dim_max - dim_min
        num_samples = (dim_range / resolution).tolist()
        num_particles = 1
        for i in range(len(num_samples)):
            num_samples[i] = int(num_samples[i])
            if num_samples[i] < 1:
                num_samples[i] = 1
            num_particles = num_particles * num_samples[i]

        dimensions = []
        for i in range(len(num_samples)):
            dimensions.append(np.linspace(dim_min[i], dim_max[i], num_samples[i]))

        grid = np.array(np.meshgrid(*dimensions)).T.reshape(-1, len(dim_min))
        return t_tensor(grid)

    def reset(self):
        self.stats = dict()

    def get_stats(self):
        return self.stats

    def draw(self, ax):
        pass

    def inference(self, obs, nuisance, gen_model, likelihood_f, slacks, params):
        """
        Method that performs approximate bayesian computations to provide the posterior distribution over
        the latent space values 'z' given the current observation 'obs'

        :param obs: Input observation.
        :param nuisance: Nuisance parameters of the generative model.
        :param gen_model: Generative model g(z,n) -> \hat{o}
        :param likelihood_f: Likelihood function used to evaluate a proposal p(g(z,n)|o)
        :param slacks: Slack values to evaluate.
        :param params: Dictionary with custom parameter values to tune the inference.
        :return: - samples: tensor with the resulting samples from the inference process (describe the posterior)
                 - likelihoods: NxM tensor with likelihood values corresponding to each sample M with slack N
                 - stats: dictionary with statistics (for statistics and debug):
                    - nevals: number of likelihood function p(\hat{o}|o) evaluations
                    - nsamples: number of sampling \hat{z} ~ p(z|z_t-1) operations
                    - ngens: number of generative \hat{z} ~ p(z|z_t-1) operations
                    - tevals: time taken by likelihood function p(\hat{o}|o) evaluations
                    - tsamples: time taken by sampling \hat{z} ~ p(z|z_t-1) operations
                    - tgens: time taken by generative \hat{z} ~ p(z|z_t-1) operations
        """

        dim_min = params["z_min"]
        dim_max = params["z_max"]
        resolution = params["resolution"]
        visualizer = params["visualizer"]

        # Generate trajectories for the gridded latent space
        tic = time.time()
        if self.trajs is None:
            # Generate samples (grid latent space)
            tic = time.time()
            self.particles = self.grid_latent_variables(dim_min, dim_max, resolution)
            self.stats["tsamples"] = time.time() - tic

            nuisance_batch = nuisance.expand(len(self.particles), len(nuisance[0]))
            trajs = gen_model.generate(z=self.particles, n=nuisance_batch)
            self.trajs = trajs.detach().cpu().numpy()
        self.stats["tgens"] = time.time() - tic

        # Compute batch likelihoods directly. Trajectories are already generated, no need to regenerate
        tic = time.time()
        likelihood = likelihood_f(obs, self.trajs, slack=slacks)
        self.stats["tevals"] = time.time() - tic

        if visualizer is not None:
            weights = likelihood.reshape(-1)
            weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))

            idx = np.argmax(weights)
            idx_slack = int(idx / len(self.particles))
            idx_part = int(idx % len(self.particles))

            draw_samples(self.particles, weights.reshape(likelihood.shape)[idx_slack], visualizer, width=resolution*0.8)

        self.stats["nevals"] = len(self.particles) * len(slacks)
        self.stats["nsamples"] = len(self.particles)
        self.stats["ngens"] = len(self.particles)

        return self.particles, likelihood.flatten(), self.stats
