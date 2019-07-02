
from common.common import *
from common.CBaseSampler import CBaseSampler


class CSamplerUniform(CBaseSampler):
    def __init__(self, params):
        super(CSamplerUniform, self).__init__()

        self.gmin = params["min"]
        self.gmax = params["max"]
        self.sampler = torch.distributions.uniform.Uniform(self.gmin, self.gmax)
        self.volume = torch.prod(self.gmax - self.gmin)
        self.sampleprob = 1 / self.volume
        self.samplelogprob = np.log(self.sampleprob)
        self.name = "CUniformSampler"

    def sample(self, nsamples, params):
        return self.sampler.sample(torch.Size([nsamples]))

    def prob(self, samples):
        """
        The probability density of a single value in a uniform distribution is 1/n where n is the volume of the interval
        """
        assert len(samples.shape) >= 2  # Check for the batch dimension

        return t_tensor(np.ones(len(samples))) * self.sampleprob

    def log_prob(self, samples):
        assert len(samples.shape) >= 2  # Check for the batch dimension

        return t_tensor(np.ones(len(samples))) * self.samplelogprob
