from common.common import *
from neural_emulators.CBaseGenerativeNeuralEmulator import CBaseGenerativeNeuralEmulator


class CGenerativeModelNeuralEmulator(CBaseGenerativeNeuralEmulator):
    def __init__(self, model):
        super(CGenerativeModelNeuralEmulator, self).__init__()

        self.model = None           # Neural Emulator Neural Network
        self.model_path = model     # Path to the saved neural network file
        self.output_dims = None
        self.input_dims = None
        self.initialize(model)
        self.NN_result = t_tensor([])

    def initialize(self, model):
        try:
            self.model = torch.load(model, map_location='cpu')
            self.output_dims = self.model.output_dim
            self.input_dims = self.model.input_dim
        except FileNotFoundError:
            print("ERROR. FILE NOT FOUND. Failed to load neural emulator model from: " + self.model_path)
            self.model = None

    def generate(self, z, n):
        if n is not None:
            assert len(z) == len(n), "Latent and nuisance batch dimension does not agree"

            zn = torch.zeros(len(z), len(z[0]) + len(n[0]))
            zn[:, 0:3] = n[:, 0:3]  # Start position is a nuisance variable in this app
            zn[:, 3:6] = z[:, 0:3]  # Goal position is the latent space of interest
            zn[:, 6:] = n[:, 3:]    # The rest of nuisance parameters are the arm controller values
        else:
            zn = z

        self.NN_result = self.model(zn)
        return self.NN_result[:, 0:int(self.output_dims)]

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


