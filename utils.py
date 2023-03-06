import numpy as np
from matplotlib import pyplot as plt


def plot_series(images):
    n = len(images)
    fig, ax = plt.subplots(1, n)
    for i in range(n):
        if len(images[i]) == 2:
            ax[i].set_title(images[i][1], loc="left")
        ax[i].plot(*(images[i][0]))
    fig.set_figwidth(16)