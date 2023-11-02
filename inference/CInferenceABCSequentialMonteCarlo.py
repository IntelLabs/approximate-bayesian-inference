import time
import numpy.random as rd

from common import CBaseInferenceAlgorithm
from utils.draw import draw_samples


class CInferenceABCSMC(CBaseInferenceAlgorithm):
    def __init__(self, gen_model, likelihood_f):
        super(CInferenceABCSMC, self).__init__(gen_model, likelihood_f)
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

    def resample(self, proposal_sampler, prior_sampler):
        idxs_reject = (self.accepted == 0).nonzero()
        idxs_accept = (self.accepted == 1).nonzero()
        if len(idxs_reject) > 0 and len(idxs_accept) > 0:
            for i in idxs_reject:
                j = rd.randint(0,len(idxs_accept))
                sample = proposal_sampler.sample(self.particles[j])
                self.particles[i] = sample
                self.nsamples_total = self.nsamples_total + 1
        else:
            for i in range(len(self.accepted)):
                sample = prior_sampler.sample(None)
                self.particles[i] = sample

    def inference(self, obs, nsamples, prior_sampler, device, visualizer=None, proposal_sampler=None,
                  alpha=t_tensor([-1E+6, -1E+5, -1E+4]), slacks=t_tensor([1E-2]), debug_objects=[],img = None, camera = None):
        assert proposal_sampler is not None
        assert self.generative_model is not None

        self.initialize(nsamples, prior_sampler)

        likelihood = np.array([])

        # While not all particles are accepted
        for threshold in alpha:
            self.accepted = torch.zeros(nsamples).double().to(self.generative_model.model.device)
            time_ini = time.time()
            timeout = 15.0
            while torch.sum(self.accepted) < nsamples and timeout > (time.time() - time_ini):
                # print("Accepted samples: ", torch.sum(self.accepted))
                # Compute acceptance of particles
                gen_obs = self.generative_model.model(self.particles).cpu().detach().numpy()  # TODO: Check how to remove the inline casts
                likelihood, grad = self.likelihood_f(obs, gen_obs, len(gen_obs), slack=slacks)

                # likelihood, grad = self.likelihood_f(self.particles, obs, self.generative_model.model, nsamples, slack=slacks)

                # Compute acceptance of particles
                self.accepted = likelihood > threshold

                # Resample not accepted particles
                self.resample(proposal_sampler, prior_sampler)

        # Compute batch weights
        likelihood = likelihood.reshape(-1)
        self.weights = likelihood

        # Organize the results based on the slack term used
        likelihood.reshape(len(slacks),-1)

        # Compute maximum likelihood particle
        idx = np.argmax(self.weights)
        idx_slack = int(idx/len(self.particles))
        idx_part = int(idx%len(self.particles))

        if visualizer is not None:
            draw_samples(self.particles, self.weights, visualizer, width=1)

        return torch.mean(self.particles,0), slacks[idx_slack], self.nsamples_total
