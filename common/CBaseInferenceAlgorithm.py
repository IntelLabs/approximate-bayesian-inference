
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT


class CBaseInferenceAlgorithm(object):
    def __init__(self):
        pass

    @staticmethod
    def get_name():
        raise NotImplementedError

    def inference(self, obs, nuisance, proposal, gen_model, likelihood_f, slacks, params):
        """
        Method that performs approximate bayesian computations to provide the posterior distribution over
        the latent space values 'z' given the current observation 'obs'

        :param obs: Input observation o.
        :param nuisance: Nuisance parameters of the generative model.
        :param gen_model: Generative model g(z,n) -> o'
        :param likelihood_f: Likelihood function used to evaluate a proposal p(g(z,n)|o)
        :param proposal: Initial proposal sample for the latent parameters. Usually sampled from the prior: z' ~ p(z)
        :param slacks: Slack values to evaluate.
        :param params: Dictionary with custom parameter values to tune the inference.
        :return: samples, likelihoods, stats
                 - samples: tensor with the resulting samples from the inference process (describe the posterior p(z|o))
                 - likelihoods: NxM tensor with likelihood values corresponding to each sample M with slack N
                 - stats: dictionary with statistics (for statistics and debug):
                    - nevals: number of likelihood function p(o'|o) evaluations
                    - nsamples: number of sampling z' ~ p(z|z_t-1) operations
                    - ngens: number of generative z' ~ p(z|z_t-1) operations
                    - tevals: time taken by likelihood function p(o'|o) evaluations
                    - tsamples: time taken by sampling z' ~ p(z|z_t-1) operations
                    - tgens: time taken by generative z' ~ p(z|z_t-1) operations
        """
        # Example of a generic algorithm
        # z' = proposal                         # Map the inputs to the notation used in the README.md
        # n = nuisance
        # o = observation
        # samples = list()
        # likelihoods = list()
        # stats = dict()
        # While inference not finished:
        #   o' = gen_model.generate(z', n)      # Compute proposal z' likelihood by evaluating the likelihood of
        #   l = likelihood_f(o, o', slacks)     # the generated observation o' using z' to the input observation o.
        #   if accept(l):                       # Determine whether to accept the sample z'.
        #      samples.append(z')
        # return samples, likelihoods, stats
        raise NotImplementedError
