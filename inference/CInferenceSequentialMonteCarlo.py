from common import CBaseInferenceAlgorithm
from utils.draw import draw_samples


class CInferenceSMC(CBaseInferenceAlgorithm):
    def __init__(self, gen_model, likelihood_f):
        super(CInferenceSMC, self).__init__(gen_model, likelihood_f)
        self.particles = None
        self.weights = None
        self.proposal_dist = None

    def initialize(self, num_particles, prior_sampler):
        self.particles = torch.zeros([num_particles, self.particle_dims]).double()
        self.weights = torch.ones(num_particles).double() / num_particles
        for i in range(num_particles):
            sample = prior_sampler.sample(None)
            self.particles[i] = sample

    def resample(self):
        assert self.proposal_dist is not None
        sample = None
        while sample is None:
            part_idx = np.random.random_integers(0, len(self.particles)-1)
            accept = np.random.rand()
            if accept > self.weights[part_idx]:
                sample = self.proposal_dist.sample(self.particles[part_idx])
        return sample

    def inference(self, obs, nsamples, prior_sampler, device, visualizer=None, proposal_sampler=None, alpha=0.4,
                  slacks=t_tensor([0.01, 0.1, 1.0]), debug_objects=[],img = None, camera = None):
        assert self.generative_model is not None
        assert proposal_sampler is not None
        self.proposal_dist = proposal_sampler

        if self.particles is None or self.weights is None:
            self.initialize(nsamples, prior_sampler)

        # Update particles with the proposal distribution
        for i in range(len(self.particles)):
            self.particles[i] = proposal_sampler.sample(self.particles[i])

        for part_idx in range(nsamples):
            # Perform the resampling step
            resample_likelihood = np.random.rand()
            if self.weights[part_idx] < np.log(resample_likelihood) or self.weights[part_idx] < alpha:
                self.particles[part_idx] = self.resample()

        # Compute batch likelihoods
        gen_obs = self.generative_model.model(self.particles).cpu().detach().numpy()  # TODO: Check how to remove the inline casts
        likelihood, grad = self.likelihood_f(obs, gen_obs, len(gen_obs), slack=slacks)

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

        return self.particles[idx_part], slacks[idx_slack], len(self.weights)
