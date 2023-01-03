import pandas as pd
import seaborn as sns
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_n_vs_speedup(data: pd.DataFrame):

    # make speedup column
    reference_data = data[data["num_processors"] == 1]
    reference_avg = reference_data.groupby("n")["time"].mean()

    data= data.merge(reference_avg.rename("ref_time"), on="n")
    data["speedup"] = data["ref_time"]/data["time"]


    sns.lineplot(x='n', y='speedup', hue='num_processors', data=data, ci=None, palette=sns.color_palette())

    # sns.catplot(data=data, x='n', y='speedup', hue='num_processors', kind="bar", ci=None)
    # sns.catplot(data=data, x='n', y='speedup', hue='num_processors', kind="box")
    # sns.catplot(data=data, x='n', y='speedup', hue='num_processors', kind="box")


    savepath = Path("plots")
    savepath.mkdir(parents=True, exist_ok=True)
    filename = str(savepath) + "/plot.png"
    # plt.savefig(filename, dpi=140)
    plt.show()

data = pd.read_csv("results/results.csv")
data = data[data["d"] == 784]
# data = data[data["n"].isin([166, 1291, 10000])]
print(data)
plot_n_vs_speedup(data)