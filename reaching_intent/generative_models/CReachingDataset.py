import json
from common.common import *
from samplers.CSamplerMultivariateNormal import CSamplerMultivariateNormal
from neural_emulators.CDataset import CDataset
from utils.misc import resample_trajectory


class CReachingDataset(CDataset):
    def __init__(self, filename, noise_sigma=0.0, dataset_sample_rate=30, output_sample_rate=30, ndims=3):
        # super(CReachingDataset, self).__init__(filename) # Parent constructor not called intentionally.
        self.filename = filename
        self.samples = list()
        self.x_samples = t_tensor([])  # Input values
        self.y_samples = t_tensor([])  # Output values
        self.dataset_load(noise_sigma, ndims, dataset_sample_rate, output_sample_rate)

    def dataset_load(self, noise_sigma=0.001, ndims=3, dataset_sample_rate=30, output_sample_rate=30,
                     prefix_samples=4, trajectory_duration=5.0):

        try:
            file = open(self.filename, 'r')
        except FileNotFoundError:
            return self.samples
        lines = file.readlines()

        # Load samples into two batched tensors of inputs and outputs
        for l in lines:
            sample = json.loads(l)

            traj_len = len(sample[1]) / 2  # Crop the stdev part of the trajectory

            out_params = resample_trajectory(t_tensor(sample[1][0:int(traj_len)]), dataset_sample_rate, output_sample_rate)

            # Input parameters are the start point of the trajectory, the endpoint and the controller parameters
            # Start point
            in_params = out_params[0:ndims]
            # End point
            in_params = torch.cat((in_params, out_params[-ndims:]))
            # Control parameters  #TODO: Recreate dataset with proper control params
            in_params = torch.cat((in_params, t_tensor([5, 0, 0.01, 0.05, 25.0])))

            if ndims * trajectory_duration * output_sample_rate > len(out_params):
                padding = out_params[-ndims:]
                while ndims * trajectory_duration * output_sample_rate > len(out_params):
                    out_params = torch.cat((out_params, padding))
            else:
                out_params = out_params[0:ndims * trajectory_duration * output_sample_rate]

            # Add optional noise to the trajectory
            if noise_sigma != 0:
                noise_cov = torch.ones_like(out_params) * noise_sigma * noise_sigma
                noise_dist = CSamplerMultivariateNormal({"mean": torch.zeros_like(out_params), "std": noise_cov})
                noise = noise_dist.sample(nsamples=1, params=None).view(-1)
                out_params = out_params + noise

            self.x_samples = torch.cat((self.x_samples, in_params.view(1, -1)))
            self.y_samples = torch.cat((self.y_samples, out_params.view(1, -1)))
            self.samples.append([in_params, out_params])

        print("Loaded %d trajectories" % len(self.x_samples))
        return self.samples
