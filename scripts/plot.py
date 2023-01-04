import pandas as pd
import seaborn as sns
from pathlib import Path
import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt

def plot_n_vs_speedup(data: pd.DataFrame):

    plot = sns.lineplot(x='n', y='speedup', hue='num_processors', data=data, errorbar=None)
    plot.set(xscale='log')

    ### Some alternatives:
    # plot = sns.catplot(data=data, x='n', y='speedup', hue='num_processors', kind="bar", errorbar=None)
    # plot = sns.catplot(data=data, x='n', y='speedup', hue='num_processors', kind="box")

    savepath = Path("plots")
    savepath.mkdir(parents=True, exist_ok=True)
    filename = str(savepath) + "/plot.png"
    # plt.savefig(filename, dpi=140)
    plt.show()


def plot_n_vs_d(data: pd.DataFrame):

    plot = sns.lineplot(x='n', y='time', hue='d', data=data, errorbar=None, 
        # palette=sns.color_palette()
        )
    plot.set(xscale='log')
    # plot.set(yscale='log')
    plot.set(xlabel="# of d-dimensional points")
    plot.set(ylabel="time (ms)")

    savepath = Path("plots")
    savepath.mkdir(parents=True, exist_ok=True)
    filename = str(savepath) + "/plot.png"
    # plt.savefig(filename, dpi=140)
    plt.show()

def plot_n_vs_d_3D(data: pd.DataFrame):
    x = np.array(data["n"])
    y = np.array(data["d"])
    z = np.array(data["time"])

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # x = np.log2(x)
    # y = np.log2(y)
    surf = ax.plot_trisurf(x, y, z,
        cmap=cm.coolwarm,
        # cmap="binary", 
        linewidth=0)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    savepath = Path("plots")
    savepath.mkdir(parents=True, exist_ok=True)
    filename = str(savepath) + "/plot.png"
    # plt.savefig(filename, dpi=140)
    plt.show()


### Import data
data = pd.read_csv("results/results.csv")

### Msake speedup column
reference_data = data[data["num_processors"] == 1]
reference_avg = reference_data.groupby(["n", "d"])["time"].mean()
data= data.merge(reference_avg.rename("ref_time"), on=["n", "d"])
data["speedup"] = data["ref_time"]/data["time"]

### Multiple runs were done with the exact same parameters. Get their average:
data_averaged = data.groupby(["n", "d", "num_processors"], as_index=False).mean(numeric_only=True)

### Do the plotting
plot_n_vs_speedup(data)
plot_n_vs_d(data[data["num_processors"] == 4])
plot_n_vs_d_3D(data_averaged[data_averaged["num_processors"] == 4])



