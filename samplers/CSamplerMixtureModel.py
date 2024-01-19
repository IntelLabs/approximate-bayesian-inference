
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT

from common.common import *
from common.CBaseSampler import CBaseSampler


class CSamplerMixtureModel(CBaseSampler):
    def __init__(self, params):
        super(CSamplerMixtureModel, self).__init__()

        self.models = params["models"]
        self.weights = params["weights"]

        assert len(self.models) == len(self.weights), \
            "len(models)=%d , len(weights)=%d" % (len(self.models), len(self.weights))

        # Make sure all weights are positive
        assert torch.sum(self.weights < 0) <= 0, "There are non-positive weights"

        # Make sure weights are normalized
        self.weights = self.weights / self.weights.sum()

        # Initialize the categorical sampler
        self.cat_sampler = torch.distributions.Categorical(probs=self.weights)

        self.name = "CSamplerMixtureModel"

    def sample(self, nsamples, params):
        # Sample from a categorical parameterized by the weights to select the model
        models = self.cat_sampler.sample(torch.Size([nsamples]))
        res = t_tensor([])

        # Sample from the model
        for m in models:
            sample = self.models[m].sample(1, None)
            res = torch.cat((res, sample))
        return res

    def log_prob(self, samples):
        assert len(samples.shape) >= 2, "Failed check for the batch dimension"

        likelihood = torch.zeros_like(samples.view(-1)[0])
        for i in range(len(self.models)):
            likelihood = likelihood + torch.exp(self.models[i].log_prob(samples)) * self.weights[i]
        return torch.log(likelihood)

    def prob(self, samples):
        assert len(samples.shape) >= 2, "Failed check for the batch dimension"

        likelihood = torch.zeros_like(samples.view(-1)[0])
        for i in range(len(self.models)):
            likelihood = likelihood + self.models[i].prob(samples) * self.weights[i]
        return likelihood
