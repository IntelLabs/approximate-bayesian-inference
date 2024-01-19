
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT

from common.common import *
from samplers.CSamplerMultivariateNormal import CSamplerMultivariateNormal
from samplers.CSamplerUniform import CSamplerUniform
from samplers.CSamplerMixtureModel import CSamplerMixtureModel


def test_sampler(sampler, nsamples):
    samples = sampler.sample(nsamples, None)
    probs = sampler.prob(samples)
    logprobs = sampler.log_prob(samples)
    for s in range(len(samples)):
        print("Generated sample with %s:" % sampler.name, samples[s].numpy(),
              "Prob: %f  LogProb: %f" % (probs[s], logprobs[s]))


if __name__ == "__main__":
    params = dict()
    models = list()
    nsamples = 10

    ######################################################################
    # TEST UNIFORM SAMPLER
    ######################################################################
    params["min"] = t_tensor([0, 0, 0])
    params["max"] = t_tensor([1, 1, 1])
    sampler = CSamplerUniform(params)
    test_sampler(sampler, nsamples)

    ######################################################################
    # TEST MULTIVARIATE SAMPLER
    ######################################################################
    params["mean"] = t_tensor([0, 0, 0])
    params["std"] = t_tensor([1, 1, 1])
    sampler = CSamplerMultivariateNormal(params)
    test_sampler(sampler, nsamples)

    ######################################################################
    # TEST MIXTURE MODEL
    ######################################################################
    models = list()
    models.append(CSamplerMultivariateNormal(params={"mean": t_tensor([0, 0, 0]), "std": t_tensor([1, 1, 1])}))
    models.append(CSamplerMultivariateNormal(params={"mean": t_tensor([0, 0, 0]), "std": t_tensor([1, 1, 1])}))
    models.append(CSamplerMultivariateNormal(params={"mean": t_tensor([0, 0, 0]), "std": t_tensor([1, 1, 1])}))
    params["models"] = models
    params["weights"] = t_tensor([0.5, 0.3, 0.2])
    sampler = CSamplerMixtureModel(params)
    test_sampler(sampler, nsamples)

