
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT


class CBaseSampler(object):
    def __init__(self):
        pass

    def sample(self, nsamples, params):
        raise NotImplementedError

    def prob(self, samples):
        raise NotImplementedError

    def log_prob(self, samples):
        raise NotImplementedError
