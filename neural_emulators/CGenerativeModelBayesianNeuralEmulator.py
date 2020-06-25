from common.common import *
from neural_emulators.CBaseGenerativeNeuralEmulator import CBaseGenerativeNeuralEmulator


class CGenerativeModelBayesianNeuralEmulator(CBaseGenerativeNeuralEmulator):
    def __init__(self, model):
        super(CGenerativeModelBayesianNeuralEmulator, self).__init__()
        self.model = None           # Neural Emulator Neural Network
        self.model_path = model     # Path to the saved neural network file
        self.output_dims = None
        self.input_dims = None
        self.initialize(model)
        self.NN_result = t_tensor([])
        self.mc_samples = 20

    @staticmethod
    def get_name():
        return "emu"

    def initialize(self, model):
        try:
            self.model = torch.load(model, map_location='cpu')
            self.output_dims = self.model.output_dim
            self.input_dims = self.model.input_dim
        except FileNotFoundError:
            print("ERROR. FILE NOT FOUND. Failed to load neural emulator model from: " + self.model_path)
            self.model = None

    def generate(self, z, n):
        self.NN_result = t_tensor(self.mc_samples, len(z), self.output_dims)
        for i in range(self.mc_samples):
            self.NN_result[i], kl = self.model(t_tensor(z))
        mu = torch.mean(self.NN_result, dim=0)
        sigma = torch.std(self.NN_result, dim=0)
        return mu[:, 0:int(self.output_dims)], sigma[:, 0:int(self.output_dims)]

    def move_to_device(self, device):
        raise NotImplementedError

    def gradient(self):
        if len(self.NN_result) > 0:
            return self.NN_result.grad
        else:
            raise Exception("Invalid gradient. Make sure to call generate() before using gradient().")

    def forward(self, x):
        raise NotImplementedError

    def num_flat_features(self, x):
        raise NotImplementedError

    def test(self, dataset):
        raise NotImplementedError

    def train(self, dataset, epochs, learning_rate):
        raise NotImplementedError


