
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT

import collections

from common import CBaseInferenceAlgorithm
from utils.CQuadtree import *
from utils.draw import draw_line
from utils.draw import get_heat_color


class CInferenceQuadtree(CBaseInferenceAlgorithm):
    def __init__(self, params=None):
        super(CInferenceQuadtree, self).__init__(params)
        self.BSPT = None
        self.particles = None
        self.weights = None

    def expand_particles(self, particles):
        new_particles = []
        particles_coords = t_tensor([])
        for p in particles:
            new_parts = self.BSPT.expand(p)
            for npart in new_parts:
                particles_coords = torch.cat((particles_coords, npart.coords))
            new_particles.extend(new_parts)
        return new_particles,particles_coords


    def inference(self, obs, device, dim_min, dim_max, resolution=0.001, slacks=t_tensor([0.01, 0.1, 1.0]),
                  EXPAND_THRESHOLD=0.5, viewer=None, debug_objects=[],img = None, camera = None):
        assert self.generative_model is not None

        time_values = collections.OrderedDict()

        tic = time.time()
        padding = t_tensor([0.2,5.0]).to(device)
        # Initialize the tree and insert initial particles
        self.BSPT = CQuadtree( dim_min[0:2], dim_max[0:2], padding )

        # Start from the root particle
        particles_to_expand = [self.BSPT.root]

        num_expansions = 0
        num_evals = 0
        time_values["init"] = time.time() - tic
        time_values["expand"] = 0.0
        # time_values["part_coords"] = 0.0
        time_values["likelihood"] = 0.0
        time_values["weights"] = 0.0
        time_values["expand_filter"] = 0.0

        while len(particles_to_expand) > 0:

            tic = time.time()
            eval_parts, particles_coords = self.expand_particles(particles_to_expand)
            time_values["expand"] += time.time() - tic

            # tic = time.time()
            num_evals = num_evals + len(eval_parts)
            # particles_coords = t_tensor([]).to(device)
            # for part in eval_parts:
            #     particles_coords = torch.cat((particles_coords, torch.cat((t_tensor(part.center).to(device),padding))))
            # time_values["part_coords"] += time.time() - tic

            # Compute likelihoods
            tic = time.time()
            gen_obs = self.generative_model.generate(particles_coords.view(-1,self.particle_dims)).cpu().detach().numpy() # TODO: Check how to remove the inline casts
            likelihood, grad = self.likelihood_f(obs, gen_obs, len(gen_obs), slack=slacks)

            time_values["likelihood"] += time.time() - tic

            # Compute batch weights.
            tic = time.time()

            # Likelihood normalization across all the slacks
            max_l = np.max(likelihood)
            likelihood = likelihood - max_l

            # By normalizing the batch-wise likelihoods
            # likelihood = likelihood.view(len(slacks),-1)
            # for i in range(len(slacks)):
            #     likelihood[i, :] = likelihood[i, :] - torch.max(likelihood[i, :])
            #     likelihood[i, :] = torch.exp(likelihood[i, :])
            #     likelihood[i, :] = (likelihood[i, :] - torch.min(likelihood[i, :])) / (torch.max(likelihood[i, :]) - torch.min(likelihood[i, :]))

            self.weights  = likelihood.reshape(-1)
            time_values["weights"] += time.time() - tic

            # Determine particles to expand
            tic = time.time()
            weights_indices = np.nonzero(self.weights > EXPAND_THRESHOLD)
            weights_indices = np.array(weights_indices) % int(len(self.weights) / len(slacks))
            part_indices = np.unique(weights_indices)
            particles_to_expand = []
            for idx, p_idx in enumerate(part_indices):
                if self.BSPT.leaves[eval_parts[p_idx].leaf_idx].radius > resolution:
                    particles_to_expand.append(self.BSPT.leaves[eval_parts[p_idx].leaf_idx])

            time_values["expand_filter"] += time.time() - tic

            # Compute maximum likelihood particle
            idx = np.argmax(self.weights)
            idx_slack = int(idx / (len(self.weights) / len(slacks)))
            idx_part = int(idx % (len(self.weights) / len(slacks)))
            part = self.BSPT.leaves[eval_parts[idx_part].leaf_idx]
            result = t_tensor(np.concatenate((part.center, padding.cpu().numpy())))
            # print("Eval %d particles. %d expansions. ML idx: %d/%d. Slack: %d" % (
            # num_evals, num_expansions, idx, len(self.weights), idx_slack))

            # Do visualization
            if viewer is not None or img is not None:
                likelihood = likelihood.reshape(len(slacks),-1)
                for i in range(len(slacks)):
                    row_min = np.min(likelihood[i, :])
                    row_range = np.max(likelihood[i, :]) - np.min(likelihood[i, :])
                    likelihood[i, :] = (likelihood[i, :] - row_min)  / row_range

                likelihood = likelihood.reshape(-1)

                for idx_part, part in enumerate(eval_parts):
                    weights = np.array([likelihood[i * len(eval_parts) + idx_part] for i in range(len(slacks))])
                    part.value = np.max(weights)
                    slack_idx = np.argmax(weights)

                for node in self.BSPT.leaves:
                    color = get_heat_color(node.value.item())
                    top_left = np.array([node.center[0] + node.radius, node.center[1] - node.radius, padding[0]])
                    top_right = np.array([node.center[0] + node.radius, node.center[1] + node.radius, padding[0]])
                    bottom_right = np.array([node.center[0] - node.radius, node.center[1] + node.radius, padding[0]])
                    bottom_left = np.array([node.center[0] - node.radius, node.center[1] - node.radius, padding[0]])

                    debug_objects.append(draw_line(top_left, top_right, color=color, width=1, lifetime=0, physicsClientId=viewer, img=img, camera=camera))
                    debug_objects.append(draw_line(top_right, bottom_right, color=color, width=1, lifetime=0, physicsClientId=viewer, img=img, camera=camera))
                    debug_objects.append(draw_line(bottom_right, bottom_left, color=color, width=1, lifetime=0, physicsClientId=viewer, img=img, camera=camera))
                    debug_objects.append(draw_line(bottom_left, top_left, color=color, width=1, lifetime=0, physicsClientId=viewer, img=img, camera=camera))
            num_expansions = num_expansions + 1

        # print(time_values)
        return result, slacks[idx_slack], num_evals * len(slacks)
