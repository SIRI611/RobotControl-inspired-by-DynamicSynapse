import os
import shutil

import numpy as np
from tqdm import tqdm

from DynamicSynapse.Utils.loggable import Loggable
from DynamicSynapse.Utils.pathselection import common_path
from DynamicSynapse.Utils.plotscripts import heatmap2d
import matplotlib.pyplot as plt
from mnist import MNIST
from DynamicSynapse.Utils.math import ReLU



class DiffWeight(Loggable):
    def __init__(self, number_of_neurons, name='DiffWeight'):
        self.number_of_neurons = number_of_neurons
        self.diff = np.zeros(self.number_of_neurons)
        self.diff_sum = np.ones(number_of_neurons)
        self.value_last = np.zeros(self.number_of_neurons)
        self.value = np.zeros(self.number_of_neurons)
        self.num_calls = 0
        self.name_list_log = ["diff"]
        self.name = name

    def step(self, value):
        self.num_calls += 1
        self.value_last = self.value
        self.value = value
        # self.diff = np.abs(self.value - self.value_last)
        self.diff = ReLU(self.value - self.value_last)
        self.diff_sum += self.diff
        self.weights = self.diff_sum/self.diff_sum.max()
        if 'debugging' in globals() and debugging:
            if self.num_calls % debug == 0:
                print('diff', self.diff)
                print('diff_sum', self.diff_sum)
        return self.weights


if __name__ == "__main__":
    from sklearn.datasets import fetch_openml

    debugging = 100
    T = 1000
    dt = 0.1
    number_of_steps = int(T / dt + 1)

    # # load dataset
    # mnist = fetch_openml('mnist_784')
    # # mnist = fetch_openml('Fashion-MNIST')
    # x_orig = mnist.data[:number_of_steps].to_numpy() / 256
    # x = x_orig  # - x_orig.mean(axis=0)
    # y = mnist.target[:number_of_steps].to_numpy()
    mndata = MNIST('F:\DataSet\MINIST', gz=True)
    images, labels = mndata.load_training()
    x = np.array(images)/256
    y = np.array(labels)
    experiment = 'InformativeSelector'
    pathes = common_path(experiment)
    for filename in os.listdir(os.getcwd()):
        if filename.endswith(".py"):
            shutil.copy2(filename, pathes['code_path'])

    IS = DiffWeight(784)
    IS.init_recording(log_path=pathes['data_path'], log_name='traces')
    for step in tqdm(range(number_of_steps)):
        pre_v = x[step]
        weight = IS.step(pre_v)
        IS.recording()
    IS.save_recording()

    heatmap2d(IS.weight.reshape(28,28))
    plt.show()
    pass