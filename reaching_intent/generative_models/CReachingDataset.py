import time
import json
from common.common import *
from samplers.CSamplerMultivariateNormal import CSamplerMultivariateNormal
from neural_emulators.CDataset import CDataset
from utils.misc import resample_trajectory

#TODO: Implement lazy-dataset load. Reserve the space but only parse the trajectory when first accessed.
#TODO: Trace and implement properly x_samples and y_samples

class CReachingDataset(CDataset):
    def __init__(self, filename, noise_sigma=0.0, dataset_sample_rate=30, output_sample_rate=30, ndims=3, n_datapoints=-1):
        # super(CReachingDataset, self).__init__(filename) # Parent constructor not called intentionally.
        self.filename = filename
        self.samples = list()
        self.x_samples = t_tensor([])  # Input values
        self.y_samples = t_tensor([])  # Output values
        self.dataset_load(noise_sigma, ndims, dataset_sample_rate, output_sample_rate, n_datapoints=n_datapoints)

    def dataset_load(self, noise_sigma=0.001, ndims=3, dataset_sample_rate=30, output_sample_rate=30,
                     prefix_samples=4, trajectory_duration=5.0, n_datapoints=-1):

        t_ini = time.time()
        print("Loading trajectories from %s" % len(self.filename))

        try:
            file = open(self.filename, 'r')
        except FileNotFoundError:
            return self.samples
        lines = list()
        if n_datapoints == -1:
            lines = file.readlines()
        else:
            for _ in range(n_datapoints):
                lines.append(file.readline())

        print("Loaded file in %.3f. Processing %d trajectories..." % ((time.time() - t_ini), len(lines)))

        # TODO: Compute and check that memory requirements

        # TODO: Prepare memory for the trajectories and avoid append and cat

        # Load samples into two batched tensors of inputs and outputs
        for idx,l in enumerate(lines):
            t_traj = time.time()
            sample = json.loads(l)

            traj_len = len(sample[1]) / 2  # Crop the stdev part of the trajectory

            out_params = resample_trajectory(t_tensor(sample[1][0:int(traj_len)]), dataset_sample_rate, output_sample_rate)

            # Input parameters are the start point of the trajectory, the endpoint and the controller parameters
            in_params = t_tensor(sample[0])

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
            self.samples.append([in_params.view(-1), out_params.view(-1)])
            if idx % 100 == 0:
                print("Progress %d / %d. Time per traj: %.3fs ETA in: %.3fs" %
                      (idx, len(lines), time.time()-t_traj, (len(lines) - idx) * (time.time()-t_traj)))

        print("Loaded %d trajectories. Took %.3f s" % (len(self.x_samples), time.time()-t_ini))
        return self.samples
