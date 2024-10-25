
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT

"""
    File name: DiscreteSpace.py
    Author: Javier Felip Leon
    Email: javier.felip.leon@intel.com
    Date created: May 2019
    Date last modified: May 2019
    Copyright: Intel Corporation
"""
from common.common import *
from common.CBaseSpace import CBaseSpace


class DiscreteSpace(CBaseSpace):
    def __init__(self, labels, sampler):
        self.labels = labels
        if sampler is None:
            sampler = torch.distributions.Categorical(torch.ones(len(labels)) * t_tensor(1/len(labels)))

        super(DiscreteSpace, self).__init__(len(labels), sampler)

    def sample(self):
        return self.labels[self.sampler.sample()]
