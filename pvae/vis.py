# visualisation helpers for data
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
sns.set()

def array_plot(points, filepath):
    data = points[0]
    period = len(points) + 1
    a = np.zeros((period*data.shape[0], data.shape[1]))
    a[period*np.array(range(data.shape[0])),:] = data
    if period > 2:
        recon = points[1]
        a[period*np.array(range(data.shape[0]))+1,:] = recon
    ax = sns.heatmap(a, linewidth=0.5, vmin=-1, vmax=1, cmap=sns.color_palette('RdBu_r', 100))
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.clf()