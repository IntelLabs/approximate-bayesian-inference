
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT

from common.common import *
from common.CBaseSampler import CBaseSampler


class CSamplerMultivariateNormal(CBaseSampler):
    def __init__(self, params):
        super(CSamplerMultivariateNormal, self).__init__()
        self.mean = params["mean"]
        self.std = params["std"]
        self.sampler = torch.distributions.MultivariateNormal(self.mean, torch.diag(self.std))
        self.name = "CNormalSampler"

    def sample(self, nsamples, params):
        return self.sampler.sample(torch.Size([nsamples]))

    def prob(self, samples):
        assert len(samples.shape) >= 2, "Failed check for the batch dimension"

        return torch.exp(self.sampler.log_prob(samples))

    def log_prob(self, samples):
        assert len(samples.shape) >= 2, "Failed check for the batch dimension"

        return self.sampler.log_prob(samples)


# Custom implementation. Some optimizations such as precomputing the determinant and the log determinant
# class CMultivariateGaussian:
#     def __init__(self, mean, cov):
#         assert len(mean) == len(cov)
#         self.mean = mean
#         self.cov = cov
#         self.cov_det = torch.potrf(cov).diag().prod()
#         self.cov_det_log = torch.log(self.cov_det)
#         self.cov_inv = cov.inverse()
#
#     def likelihood(self, data):
#         assert len(data) == len(self.mean)
#
#         k = len(data)
#
#         term1 = torch.DoubleTensor([-(k/2) * math.log(2 * math.pi)]).to(data.device)
#
#         term2 = -0.5 * self.cov_det_log
#
#         diff = data - self.mean
#
#         # term3 = -0.5 * diff.view(1, -1) @ self.cov_inv @ diff.view(-1, 1) #Python >= 3.5
#
#         term3 = -0.5 * torch.matmul(diff.view(1, -1), torch.matmul(self.cov_inv, diff.view(-1, 1)))
#
#         log_likelihood = term1 + term2 + term3
#
#         return torch.exp(log_likelihood)


if __name__ == "__main__":
    params = dict()
    params["mean"] = t_tensor([0, 0, 0])
    params["std"] = t_tensor([1, 1, 1])
    sampler = CSamplerMultivariateNormal(params)
    samples = sampler.sample(10, None)
    probs = sampler.prob(samples)
    logprobs = sampler.log_prob(samples)
    for s in range(len(samples)):
        print("Generated sample with %s:" % sampler.name, samples[s].numpy(),
              "Prob: %f  LogProb: %f" % (probs[s], logprobs[s]))


