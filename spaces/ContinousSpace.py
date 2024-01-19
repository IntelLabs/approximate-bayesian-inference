
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT

"""
    File name: ContinousSpace.py
    Author: Javier Felip Leon
    Email: javier.felip.leon@intel.com
    Date created: May 2019
    Date last modified: May 2019
    Copyright: Intel Corporation
"""
from common.common import *
from common.CBaseSpace import CBaseSpace


class ContinousSpace(CBaseSpace):
    def __init__(self, dims, sampler=None, min=None, max=None):

        self.min = min
        self.max = max

        if sampler is None:
            sampler = torch.distributions.MultivariateNormal(torch.zeros(dims), torch.eye(dims))
        super(ContinousSpace, self).__init__(dims, sampler)

    def sample(self):
        sample = self.sampler.sample()

        if self.max is not None:
            sample = torch.min(self.max, sample)

        if self.min is not None:
            sample = torch.max(self.min, sample)
        return sample
