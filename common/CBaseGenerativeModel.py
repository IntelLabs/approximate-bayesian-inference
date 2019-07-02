"""
    File name: CGenerativeModelBase.py
    Author: Javier Felip Leon
    Email: javier.felip.leon@intel.com
    Date created: Jun 2019
    Date last modified: Jun 2019
    Copyright: Intel Corporation
"""


class CBaseGenerativeModel:
    def initialize(self, model):
        raise NotImplementedError

    def generate(self, z, n):
        raise NotImplementedError

    class Model(object):
        def __init__(self, generate_f, input_dim, output_dim, device):
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.generate_f = generate_f
            self.device = device

        def __call__(self, params):
            return self.generate_f(params)
