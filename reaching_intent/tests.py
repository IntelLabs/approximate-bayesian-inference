import matplotlib.pyplot as plt
import torch
import math
import numpy as np

dims = 100
mean_err = 0.02
stdevs = np.arange(0.001, 10.0, 0.001)

err_term = []
log_term = []

for stdev in stdevs:
    stdev_inv = 1/stdev

    variances_inv = torch.Tensor([stdev_inv*stdev_inv]*dims)

    log_term.append(-torch.sum(torch.log(variances_inv)))

    err_term.append((mean_err * torch.diag(variances_inv) * mean_err).sum())

err = np.array(log_term) + np.array(err_term)

plt.axvline(x=stdevs[err.argmin()])
plt.plot(stdevs, err_term, label="Error term")
plt.plot(stdevs, log_term, label="Log term")
plt.plot(stdevs, err, label="Loss")
plt.xlabel("sigma")
plt.ylabel("error")
plt.legend()
plt.show()
