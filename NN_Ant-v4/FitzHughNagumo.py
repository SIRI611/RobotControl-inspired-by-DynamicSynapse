#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :FitzHughNagumo.py
# @Time      :2022/12/2 21:30
# @Author    :Siri
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import copy


def rk4(h, y, inputs, f):
    k1 = f(y, inputs)
    k2 = f(y + 0.5 * h * k1, inputs)
    k3 = f(y + 0.5 * h * k2, inputs)
    k4 = f(y + k3 * h, inputs)
    return y + (k1 + 2 * (k2 + k3) + k4) * h / 6.0

class FHNN:
    def __init__(self, NumberOfNeurons, a=None, b=None, c=None, I=None, V=None, W=None, t=0, scale=1):
        self.a = a * np.ones(NumberOfNeurons) if a is not None else 0.08 * np.ones(NumberOfNeurons)
        self.b = b * np.ones(NumberOfNeurons) if b is not None else 2 * np.ones(NumberOfNeurons)
        self.c = c * np.ones(NumberOfNeurons) if c is not None else 0.8 * np.ones(NumberOfNeurons)
        self.I = I * np.ones(NumberOfNeurons) if I is not None else 0 * np.ones(NumberOfNeurons)
        self.Vn = V * np.ones(NumberOfNeurons) if V is not None else 0 * np.ones(NumberOfNeurons)
        self.Wn = W * np.ones(NumberOfNeurons) if W is not None else 0 * np.ones(NumberOfNeurons)
        self.Vp = self.V * np.ones(NumberOfNeurons) if not V is None else 0 * np.ones(NumberOfNeurons)
        self.Wp = self.W * np.ones(NumberOfNeurons) if not W is None else 0 * np.ones(NumberOfNeurons)
        self.t = t
        self.scale = scale

    def Derivative(self, state, inputs, NeuronID=None):
        V, W = state
        I = inputs
        Dv = (V - np.power(V, 3) - W + I) * self.scale
        if NeuronID == None:
            DW = (self.a * (self.b * V - self.c * W)) * self.scale
        else:
            DW = (self.a[NeuronID] * (self.b[NeuronID] * V - self.c[NeuronID] * W)) * self.scale
        return np.array([Dv, DW])

    def StepDynamics(self, dt, I):
        self.t += dt
        self.I = I
        self.Vn, self.Wn = rk4(dt, [self.Vp, self.Wp], self.I, self.Derivative)
        assert np.logical_not(
            np.logical_or(np.any(np.isnan([self.Vn, self.Vp])), np.any(np.isinf([self.Vn, self.Vp])))), \
            "\nself.Vn=" + str(self.Vn) \
            + "\nself.Wn=" + str(self.Wn) \
            + "\nself.Vp=" + str(self.Vp) \
            + "\nself.Wp=" + str(self.Wp) \
            + "\nself.I=" + str(self.I)
        self.Vn[self.Vn > 2] = 2
        self.Vn[self.Vn < -2] = -2
        self.Wn[self.Wn > 2] = 2
        self.Wn[self.Wn < -2] = -2
        return [self.Vn, self.Wn]

    def Update(self):
        assert np.logical_not(
            np.logical_or(np.any(np.isnan([self.Vn, self.Vp])), np.any(np.isinf([self.Vn, self.Vp])))), \
            "\nself.Vn=" + str(self.Vn) \
            + "\nself.Wn=" + str(self.Wn) \
            + "\nself.Vp=" + str(self.Vp) \
            + "\nself.Wp=" + str(self.Wp)

        self.Vp, self.Wp = self.Vn, self.Wn

    def UpdateParameters(self, Parameters):
        Parameters = np.array(Parameters)
        self.scale = Parameters[:, 0]
        self.a = Parameters[:, 1]
        self.b = Parameters[:, 2]
        self.c = Parameters[:, 3]

    def InitRecording(self):
        self.Trace = {
            'Vn': deque(),
            'Wn': deque(),
            't': deque(),
            'I': deque()
        }

    def Recording(self, StepNumber,number):
        for key in self.Trace:
            exec("self.Trace['%s'].append(copy.deepcopy(self.%s))" % (key, key))
        # self.Trace["StepNumber"] = StepNumber
        # self.Trace['NumberOfLayer'] =  number,


if __name__ == "__main__":
    run_code = 0
