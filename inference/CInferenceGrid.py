from common import CBaseInferenceAlgorithm
from utils.draw import draw_samples


class CInferenceGrid(CBaseInferenceAlgorithm):
    def __init__(self, params=None):
        super(CInferenceGrid, self).__init__(params)
        self.particles = None
        self.weights = None
        self.proposal_dist = None
        self.resolution = 0

    def initialize(self, resolution, dimension_min, dimension_max):
        dim_range = dimension_max - dimension_min
        num_samples = (dim_range / resolution).tolist()
        num_particles = 1
        self.resolution = resolution
        for i in range(len(num_samples)):
            num_samples[i] = int(num_samples[i])
            if num_samples[i] < 1:
                num_samples[i] = 1
            num_particles = num_particles * num_samples[i]
        self.particles = torch.zeros([num_particles, self.particle_dims]).double()
        self.weights = torch.ones(num_particles).double() / num_particles

        dimensions = []
        for i in range(len(num_samples)):
            dimensions.append(np.linspace(dimension_min[i], dimension_max[i], num_samples[i]))

        particles = np.array(np.meshgrid(*dimensions)).T.reshape(-1,self.particle_dims)

        self.particles = t_tensor(particles)
        trajs = self.generative_model.generate(self.particles)
        self.grid_trajs = trajs.detach().cpu().numpy()

    def inference(self, obs, device, dim_min, dim_max, resolution=0.05, slacks=t_tensor([0.01, 0.1, 1.0]),
                  visualizer = None, debug_objects=[],img = None, camera = None):

        assert self.generative_model is not None
        if self.particles is None or self.weights is None or self.resolution != resolution:
            self.initialize(resolution, dim_min, dim_max)
            print("Initialize grid!")

        # Compute batch likelihoods directly. Trajectories are already generated, no need to regenerate
        likelihood, grad = self.likelihood_f(obs, self.grid_trajs, len(self.particles), slack=slacks)

        # Compute batch weights
        likelihood = likelihood.reshape(-1)
        # self.weights = likelihood
        self.weights = (likelihood - np.min(likelihood)) / (np.max(likelihood) - np.min(likelihood))
        # self.weights = torch.exp(self.weights)
        # self.weights = (self.weights - torch.min(self.weights)) / (torch.max(self.weights) - torch.min(self.weights))

        # Compute maximum likelihood particle
        idx = np.argmax(self.weights)
        idx_slack = int(idx/len(self.particles))
        idx_part = int(idx%len(self.particles))

        if visualizer is not None:
            draw_samples(self.particles, self.weights, visualizer, width=5)

        return self.particles[idx_part], slacks[idx_slack], len(self.weights)
