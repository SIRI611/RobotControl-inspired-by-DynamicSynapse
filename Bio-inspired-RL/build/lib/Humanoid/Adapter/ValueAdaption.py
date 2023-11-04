# -*- coding: utf-8 -*-
"""
Created on Sat Apr 07 12:31:37 2018

@author: chitianqilin
"""
import numpy as np

from DynamicSynapse.Utils.loggable import Loggable


class WeightAdapter(Loggable):
    def __init__(self, number_of_weights, inital_weights=1, update_rate=0.0001, target_average=1, direction='both'):
        # if inital_weights is not None:
        #     assert inital_weights.shape == number_of_weights, \
        #         "inital_weights has a different shape from number_of_weights"
        #     self.weights = inital_weights
        # else:
        #     self.weights = np.ones(number_of_weights)
        self.weights = np.ones(number_of_weights) * inital_weights
        self.update_rate = np.ones(number_of_weights) * update_rate
        self.target_average = np.ones(number_of_weights) * target_average
        self.weights_var = np.zeros(number_of_weights)
        self.direction = direction
        self.name_list_log = ["weights"]
        self.name = "WeightAdapter"

    def step(self, dt, input):
        self.weights_var= (self.target_average - input*self.weights) * self.update_rate * dt
        if self.direction == "both":
             self.weights += self.weights_var
        elif self.direction == "numb":
            self.weights_var[np.greater(self.weights_var, 0)] = 0
            self.weights += self.weights_var
        elif self.direction == "sensitive":
            self.weights_var[np.less(self.weights_var, 0)] = 0
            self.weights += self.weights_var
        return self.weights