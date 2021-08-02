import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import sort
from collections import Counter
import os
import sys


class PlotUtil:
    def __init__(
        self,
        relation="",
        file_dir="",
        dir_suffix="",
    ):
        super(PlotUtil, self).__init__()
        self.fig, self.ax = self.initial(relation)
        self.file_dir = "{}/{}".format(file_dir, dir_suffix)
        self.file_path = "{}/{}.png".format(self.file_dir, relation)
        # self.filename="{}/{}.png".format(self.plot_save_path, relation)

    def initial(self, relation):
        fig, ax = plt.subplots()
        ax.set(xlabel="# Entities", ylabel="Normalized score", title=relation)
        return fig, ax

    def draw_num_entity_to_score(self, arr):
        arr = np.sort(arr)
        arr = normalization(arr)
        result = Counter(arr)
        x = list(result.values())
        y = list(result.keys())
        cumsum_x = np.cumsum(x, axis=0)
        # self.ax.plot(x, y)
        self.ax.plot(cumsum_x, y)
        x=range(len(arr))
        self.ax.plot(x,y)
        # plt.show()

    def show(self):
        plt.show()

    def save(self):
        if not os.path.exists(self.file_dir):
            os.makedirs(self.file_dir)
        self.fig.tight_layout()
        self.fig.savefig(self.file_path)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


if __name__ == "__main__":
    np.random.seed(123)
    arr = np.random.randint(0, 100, size=(100,))
    plot_util = PlotUtil()
    plot_util.draw_num_entity_to_score(arr)
    plot_util.show()
