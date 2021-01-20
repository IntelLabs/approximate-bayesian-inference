
import numpy as np
from matplotlib import pyplot as plt

files = ["/home/jfelip/workspace/approximate-bayesian-inference/reaching_intent/pytorch_models/ne_fc4_10k_MSE_in11_out450.pt.train_report.txt",
         "/home/jfelip/workspace/approximate-bayesian-inference/reaching_intent/pytorch_models/ne_fc3_10k_MSE_in11_out450.pt.train_report.txt"]

for file in files:
    data = np.loadtxt(file)
    plt.plot(data[:, 1], label="train_%s" % file.split("/")[-1])
    plt.plot(data[:, 2], label="test_%s" % file.split("/")[-1])

plt.legend()
plt.show()
