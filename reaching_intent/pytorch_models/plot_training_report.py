
import numpy as np
from matplotlib import pyplot as plt

file = "/home/labuser/workspace/approximate-bayesian-inference/reaching_intent/pytorch_models/ne_bfc4_10k_MSE_in11_out450.pt.train_report.txt"

data = np.loadtxt(file)

plt.plot(data[:, 0], data[:, 1], label="train")
plt.plot(data[:, 0], data[:, 2], label="test")
plt.legend()
plt.show()
