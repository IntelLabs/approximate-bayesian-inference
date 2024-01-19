
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT

import json
import torch.utils.data
from common.common import *


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
