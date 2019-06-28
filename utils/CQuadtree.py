import numpy as np
import matplotlib.pyplot as plt
import time
import torch

class CNode:
    def __init__(self, center, idx, radius, padding, level):
        self.center = center
        self.radius = radius
        self.children = [None,None,None,None]
        self.value = torch.DoubleTensor([0])
        self.leaf_idx = idx
        self.coords = torch.cat((torch.DoubleTensor(center), torch.DoubleTensor([padding])))
        self.level = level

    def __lt__(self, other):
        return self.value < other.value

    def is_leaf(self):
        return self.leaf_idx != -1

    def __float__(self):
        return self.value

    def __repr__(self):
        return "[ " + str(self.leaf_idx) +" , "+ str(self.radius) +  " ]"


class CQuadtree:
    def __init__(self, dmin, dmax, padding):
        self.root = CNode(center=(dmin + dmax) / 2, idx=0, radius=torch.max((dmax - dmin) / 2), padding=padding, level=0)
        self.nodes = [self.root]
        self.leaves = [self.root]
        self.padding = padding

    def expand(self, particle):
        if not particle.is_leaf():
            return []
        p1_center = [particle.center[0] + particle.radius/2, particle.center[1] + particle.radius/2]
        p2_center = [particle.center[0] + particle.radius/2, particle.center[1] - particle.radius/2]
        p3_center = [particle.center[0] - particle.radius/2, particle.center[1] - particle.radius/2]
        p4_center = [particle.center[0] - particle.radius/2, particle.center[1] + particle.radius/2]
        p1 = CNode(center=p1_center, idx=particle.leaf_idx, radius=particle.radius/2, padding=self.padding, level=particle.level+1)
        p2 = CNode(center=p2_center, idx=len(self.leaves),  radius=particle.radius/2, padding=self.padding, level=particle.level+1)
        p3 = CNode(center=p3_center, idx=len(self.leaves)+1,radius=particle.radius/2, padding=self.padding, level=particle.level+1)
        p4 = CNode(center=p4_center, idx=len(self.leaves)+2,radius=particle.radius/2, padding=self.padding, level=particle.level+1)

        self.leaves[particle.leaf_idx] = p1
        self.leaves.append(p2)
        self.leaves.append(p3)
        self.leaves.append(p4)
        self.nodes.append(p1)
        self.nodes.append(p2)
        self.nodes.append(p3)
        self.nodes.append(p4)
        particle.children = [p1, p2, p3, p4]
        particle.leaf_idx = -1

        return particle.children


if __name__ == '__main__':
    dmin = np.array([-1, -0.2])
    dmax = np.array([1, 0.7])
    num_splits = 100

    ini = time.time()
    tree = CQuadtree(dmin, dmax)
    for i in range(num_splits):
        if len(tree.leaves) > 1:
            idx = np.random.randint(0,len(tree.leaves)-1)
        else:
            idx = 0
        tree.expand(tree.leaves[idx])

    print("Quadtree creation and %d splts. Took: %fs" % (num_splits,time.time()-ini))

    fig, ax = plt.subplots()
    plt.ion()
    plt.show()

    lim_min = np.min(tree.root.center - tree.root.radius)
    lim_max = np.max(tree.root.center + tree.root.radius)
    plt.xlim(lim_min - 0.1, lim_max + 0.1)
    plt.ylim(lim_min - 0.1, lim_max + 0.1)

    while plt.get_fignums():
        idx = np.random.randint(0, len(tree.leaves) - 1)
        tree.expand(tree.leaves[idx])
        last_drawn = 0
        plt.cla()
        for idx in range(last_drawn, len(tree.nodes)-1):
            n = tree.nodes[idx]
            shapes =[]
            shapes.append(plt.Line2D((n.center[0] + n.radius, n.center[0] + n.radius), (n.center[1] + n.radius, n.center[1] - n.radius), lw=1))
            shapes.append(plt.Line2D((n.center[0] + n.radius, n.center[0] - n.radius), (n.center[1] - n.radius, n.center[1] - n.radius), lw=1))
            shapes.append(plt.Line2D((n.center[0] - n.radius, n.center[0] - n.radius), (n.center[1] - n.radius, n.center[1] + n.radius), lw=1))
            shapes.append(plt.Line2D((n.center[0] - n.radius, n.center[0] + n.radius), (n.center[1] + n.radius, n.center[1] + n.radius), lw=1))
            last_drawn = idx

            for s in shapes:
                ax.add_artist(s)
        plt.pause(0.01)