from common.common import *
from common.CBaseInferenceAlgorithm import CBaseInferenceAlgorithm
from utils.draw import draw_samples


class CInferenceGrid(CBaseInferenceAlgorithm):
    def __init__(self):
        super(CInferenceGrid, self).__init__()

    @staticmethod
    def grid_latent_variables(dim_min, dim_max, resolution):
        dim_range = dim_max - dim_min
        num_samples = (dim_range / resolution).tolist()
        num_particles = 1
        for i in range(len(num_samples)):
            num_samples[i] = int(num_samples[i])
            if num_samples[i] < 1:
                num_samples[i] = 1
            num_particles = num_particles * num_samples[i]

        dimensions = []
        for i in range(len(num_samples)):
            dimensions.append(np.linspace(dim_min[i], dim_max[i], num_samples[i]))

        grid = np.array(np.meshgrid(*dimensions)).T.reshape(-1, len(dim_min))
        return t_tensor(grid)

    def inference(self, obs, nuisance, proposal, gen_model, likelihood_f, params, device=torch.device("cpu"), visualizer=None,
                  slacks=t_tensor([1E-4]), img=None, camera=None):

        dim_min = params["z_min"]
        dim_max = params["z_max"]
        resolution = params["resolution"]

        particles = self.grid_latent_variables(dim_min, dim_max, resolution)
        # weights = torch.ones(len(particles)) / len(particles)

        # Generate trajectories for the gridded latent space
        nuisance_batch = nuisance.expand(len(particles), len(nuisance[0]))
        trajs = gen_model.generate(z=particles, n=nuisance_batch)
        grid_trajs = trajs.detach().cpu().numpy()

        # Compute batch likelihoods directly. Trajectories are already generated, no need to regenerate
        likelihood, grad = likelihood_f(obs, grid_trajs, len(particles), slack=slacks)

        # Compute batch weights
        likelihood = likelihood.reshape(-1)
        # self.weights = likelihood
        weights = (likelihood - np.min(likelihood)) / (np.max(likelihood) - np.min(likelihood))
        # self.weights = torch.exp(self.weights)
        # self.weights = (self.weights - torch.min(self.weights)) / (torch.max(self.weights) - torch.min(self.weights))

        # Compute maximum likelihood particle
        idx = np.argmax(weights)
        idx_slack = int(idx/len(particles))
        idx_part = int(idx%len(particles))

        if visualizer is not None:
            draw_samples(particles, weights, visualizer, width=resolution*0.8)

        return particles[idx_part], slacks[idx_slack], len(weights)
