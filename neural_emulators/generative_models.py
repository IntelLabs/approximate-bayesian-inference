from neural_emulators.common import *
import torch.nn as nn
import torch.utils.data
import json

class CGenerativeModel:
    def initialize(self, model):
        raise NotImplementedError

    def generate(self, params):
        raise NotImplementedError

    def gradient(self):
        raise NotImplementedError


class CGenerativeModelNN(nn.Module):
    def __init__(self):
        super(CGenerativeModelNN, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def num_flat_features(self, x):
        raise NotImplementedError

    def test(self, dataset):
        raise NotImplementedError

    def train(self, dataset, epochs, learning_rate):
        raise NotImplementedError


class CGenerativeModelNeuralEmulator(CGenerativeModel):
    def __init__(self, model):
        self.initialize(model)
        self.NN_result = t_tensor([])

    def initialize(self, model):
        try:
            self.model = torch.load(model, map_location='cpu')
            self.output_dims = self.model.output_dim
            self.input_dims = self.model.input_dim
        except FileNotFoundError:
            self.model = 0

    def generate(self, params):
        self.NN_result = self.model(params.double())
        return self.NN_result

    def move_to_device(self, device):
        raise NotImplementedError()

    def gradient(self):
        if len(self.NN_result) > 0:
            return self.NN_result.grad
        else:
            raise Exception("Invalid gradient. Make sure to call generate() before using gradient().")


class CDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.samples = []
        self.dataset_load(filename)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def dataset_load(self, filename):
        try:
            file = open(filename, 'r')
        except FileNotFoundError:
            return self.samples
        lines = file.readlines()

        # Load samples into a list
        for l in lines:
            sample = json.loads(l)
            in_params  = t_tensor(sample[0])
            out_params = t_tensor(sample[1])
            self.samples.append([in_params, out_params])

        print("Loaded %d samples" % len(self.samples))
        return self.samples

    def dataset_save(self, filename):
        file = open(filename, 'a')
        out = [None, None]
        for d in self.samples:
            out[0] = d[0].detach().tolist()
            out[1] = d[1].detach().tolist()
            json.dump(out, file)
            file.write('\n')
        file.close()
