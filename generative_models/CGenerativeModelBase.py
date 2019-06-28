"""
    File name: CGenerativeModelBase.py
    Author: Javier Felip Leon
    Email: javier.felip.leon@intel.com
    Date created: Jun 2019
    Date last modified: Jun 2019
    Copyright: Intel Corporation
"""
from common import *


class CGenerativeModel:
    def initialize(self, model):
        raise NotImplementedError

    def generate(self, z, n):
        raise NotImplementedError
