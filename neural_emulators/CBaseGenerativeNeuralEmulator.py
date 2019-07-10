"""
This class extends the generic generative model with the methods that a neural emulator must implement

"""
from common.CBaseGenerativeModel import CBaseGenerativeModel


class CBaseGenerativeNeuralEmulator(CBaseGenerativeModel):
    def __init__(self):
        super(CBaseGenerativeNeuralEmulator, self).__init__()

    def initialize(self, model):
        raise NotImplementedError

    def generate(self, z, n):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def num_flat_features(self, x):
        raise NotImplementedError

    def test(self, dataset):
        raise NotImplementedError

    def train(self, dataset, epochs, learning_rate):
        raise NotImplementedError

    @staticmethod
    def get_name():
        raise NotImplementedError

