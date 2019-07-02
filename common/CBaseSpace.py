"""
    File name: BaseSpace.py
    Author: Javier Felip Leon
    Email: javier.felip.leon@intel.com
    Date created: May 2019
    Date last modified: May 2019
    Copyright: Intel Corporation
"""


class CBaseSpace(object):
    def __init__(self, ndims, sampler):
        self.dimensions = ndims
        self.sampler = sampler

    def sample(self):
        return self.sampler.sample()


