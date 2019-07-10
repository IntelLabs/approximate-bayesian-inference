import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from itertools import cycle

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

caption = "Pattern-filled ellipses represent inference methods using neural likelihood surrogate. Ellipses show 1-stdev boundary after averaging 30 experiments."
title = "Error vs. Average time per frame."
x_label = "Time per frame (ms)"
y_label = "Error (m)"

# markers = [".", "v", "^", "<", ">", "o", "2",  "3", "4"]
markers = ["o"]
markercycler = cycle(markers)
linestyle = "-"

method = ["surrogate_pf (300 samples)","surrogate_bspt ($\\rho=0.005$)","surrogate_abc-rej (300 samples)","surrogate_abc-smc (300 samples)","surrogate_mcmc (100 samples, 50 burn-in)","simulator_pf (50 samples)", "simulator_bspt ($\\rho=0.01$)","simulator_abc-rej (50 samples)","simulator_abc-smc (50 samples)","simulator_mcmc (50 samples)"]
x =      [30.079,   16.479,           912.577,      138.977,      449.728,   8499.625, 13037.949,        47514.167,    46972.884, 39510.307] # Avg frame time
x_std =  [2.302,    16.435,           825.975,      275.085,      424.190,   831.767,  10244.148,        11072.989,    29121.750, 3282.331]
y =      [0.071,    0.048,            0.091,        0.106,        0.065,     0.238,    0.114,            0.191,        0.207,     0.142]   # Avg error
y_std =  [0.090,    0.078,            0.118,        0.112,        0.109,     0.1,      0.142,            0.098,        0.117,     0.122]
emu =    [True,     True,             True,         True,         True,      False,    False,            False,        False,      False]


# x = np.log(np.array(x))
# x_std = np.log(np.array(x_std))

alpha = 0.3
color = [ [1, 0, 0, 1],
          [0, 1, 0, 1],
          [0, 0, 1, 1],
          [0, 1, 1, 1],
          [1, 0, 1, 1],

          [1, 0, 0, 1],
          [0, 1, 0, 1],
          [0, 0, 1, 1],
          [0, 1, 1, 1],
          [1, 0, 1, 1]]

linecolor = [0,0,0,1]
markercolor = [0.3,0.3,0.3,1]

ax = plt.subplot(111)
ells = []
for i in range(len(x)):
    if emu[i]:
        ell = Ellipse(xy=(x[i], y[i]), width=x_std[i], height=y_std[i],
                      angle=0, edgecolor=linecolor, fill=True, facecolor=color[i], alpha=alpha, hatch='xx', clip_box=ax.bbox)
    else:
        ell = Ellipse(xy=(x[i], y[i]), width=x_std[i], height=y_std[i],
                      angle=0, edgecolor=linecolor, fill=True, facecolor=color[i], alpha=alpha)
    ells.append(ell)
    ax.add_artist(ell)
    plt.scatter(x[i], y[i], label=method[i], color=markercolor, s=100)

legend_font = matplotlib.font_manager.FontProperties(family="monospace", style=None, variant=None, weight=None,
                                                      stretch=None, size=22, fname=None, _init=None)
# ax.legend(prop=lengend_font, loc='upper left')
ax.legend(ells, [method[i] for i in range(len(method))], loc='upper left', prop=legend_font)
# ax.legend(loc='upper left')
ax.set_title(title, fontsize=28)
ax.grid(True)
plt.xlabel(x_label, fontsize=22)
plt.ylabel(y_label, fontsize=22)
plt.xlim(1, np.max(np.array(x) + np.array(x_std)))
plt.ylim(0, np.max(np.array(y) + np.array(y_std)))
ax.set_xscale("log")
plt.show()