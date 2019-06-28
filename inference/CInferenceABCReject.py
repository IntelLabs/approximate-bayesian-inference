import time

from common import *
from inference import CBaseInferenceAlgorithm
from utils.draw import draw_samples


class CInferenceABCReject(CBaseInferenceAlgorithm):
    def __init__(self, gen_model, likelihood_f):
        super(CInferenceABCReject, self).__init__(gen_model, likelihood_f)
        self.particles = None
        self.weights = None
        self.accepted = None
        self.nsamples_total = 0

    def initialize(self, num_particles, prior_sampler):
        self.nsamples_total = num_particles
        self.particles = torch.zeros([num_particles, self.particle_dims]).double()
        self.weights = torch.ones(num_particles).double() / num_particles
        self.accepted = torch.zeros(num_particles).double()
        for i in range(num_particles):
            sample = prior_sampler.sample(None)
            self.particles[i] = sample

    def resample(self, prior_sampler):
        idxs = (self.accepted == 0).nonzero()
        for i in idxs:
            sample = prior_sampler.sample(None)
            self.particles[i] = sample
            self.nsamples_total = self.nsamples_total + 1

    def inference(self, obs, nsamples, prior_sampler, device, visualizer=None, proposal_sampler=None,
                  alpha=t_tensor([-1E+6]), slacks=t_tensor([1E-3]), debug_objects=[],img = None, camera = None):
        assert self.generative_model is not None

        self.initialize(nsamples, prior_sampler)

        # While not all particles are accepted
        timeout = 10.0
        time_ini = time.time()
        while torch.sum(self.accepted) < nsamples and timeout > (time.time() - time_ini):
            # Compute acceptance of particles
            gen_obs = self.generative_model.model(self.particles).cpu().detach().numpy()  # TODO: Check how to remove the inline casts
            likelihood, grad = self.likelihood_f(obs, gen_obs, len(gen_obs), slack=slacks)

            # Compute acceptance of particles
            self.accepted = likelihood > alpha

            # Resample not accepted particles
            self.resample(prior_sampler)

            # Print number of accepted samples in one iteration
            # print("ABC-Reject. Threshold", alpha ,"Accepted samples:", torch.sum(self.accepted))


        # Compute batch weights
        likelihood = likelihood.reshape(-1)
        self.weights = likelihood

        # Organize the results based on the slack term used
        likelihood.reshape(len(slacks),-1)

        idx = np.argmax(self.weights)
        idx_slack = int(idx/len(self.particles))
        idx_part = int(idx%len(self.particles))

        if visualizer is not None:
            draw_samples(self.particles, self.weights, visualizer, width=1)

        # Return expected value
        return torch.mean(self.particles,0), slacks[idx_slack], self.nsamples_total
