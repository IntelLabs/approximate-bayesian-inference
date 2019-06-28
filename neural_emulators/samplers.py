import torch
import math

class CUniformSampler:
    def __init__(self, min, max):
        self.gmin = min
        self.gmax = max
        self.sampler = torch.distributions.uniform.Uniform(min, max)

    def sample(self, params):
        return self.sampler.sample()


class CNormalSampler:
    def __init__(self, mu, sigma):
        self.mean = mu
        self.std  = sigma
        self.sampler = torch.distributions.normal.Normal(mu, sigma)

    def sample(self, params):
        return self.sampler.sample()


class CBoundedNormalSampler:
    def __init__(self, min, max, mu, sigma):
        self.mean = mu
        self.std  = sigma
        self.min = min
        self.max = max
        self.sampler = torch.distributions.normal.Normal(mu, sigma)

    def sample(self, params):
        sample = self.sampler.sample().to(params.device)
        params = params + sample.view(params.shape)
        params = torch.max(params, self.min.to(params.device))
        params = torch.min(params, self.max.to(params.device))
        return params


class CMultivariateGaussian:
    def __init__(self, mean, cov):
        assert len(mean) == len(cov)
        self.mean = mean
        self.cov = cov
        self.cov_det = torch.potrf(cov).diag().prod()
        self.cov_det_log = torch.log(self.cov_det)
        self.cov_inv = cov.inverse()

    def likelihood(self, data):
        assert len(data) == len(self.mean)

        k = len(data)

        term1 = torch.DoubleTensor([-(k/2) * math.log(2 * math.pi)]).to(data.device)

        term2 = -0.5 * self.cov_det_log

        diff = data - self.mean

        # term3 = -0.5 * diff.view(1, -1) @ self.cov_inv @ diff.view(-1, 1) #Python >= 3.5

        term3 = -0.5 * torch.matmul(diff.view(1, -1), torch.matmul(self.cov_inv, diff.view(-1, 1)))

        log_likelihood = term1 + term2 + term3

        return torch.exp(log_likelihood)


class CMixtureModel:
    def __init__(self, models, weights):

        assert len(models) == len(weights), "len(models)=%d , len(weights)=%d" % (len(models), len(weights))

        # Make sure all weights are positive
        assert torch.sum(weights < 0) <= 0, "There are non-positive weights"

        # Make sure weights are normalized
        weights = weights / weights.sum()

        self.models = models
        self.weights = weights

    def log_prob(self, data):
        likelihood = torch.zeros_like(data.view(-1)[0])
        for i in range( len(self.models) ):
            likelihood = likelihood + torch.exp(self.models[i].log_prob(data)) * self.weights[i]
        return likelihood
