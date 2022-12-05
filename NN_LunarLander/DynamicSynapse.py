#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :DynamicSynapse.py
# @Time      :2022/11/28 9:59
# @Author    :Siri
import Functions
import numpy as np
import copy
from collections import deque
import json
import pandas as pd
import matplotlib.pyplot as plt


class DynamicSynapseArray:
    def __init__(self, NumberOfSynapses = [1, 3], Period = None, tInPeriod = None, PeriodVar = None, Amp=None,
                 WeightersCentre=None, WeightersCentreUpdateRate=0.000012, WeightersOscilateDecay=0.0000003,
                 ModulatorAmount=0, InitAmp=0.4, t=0, dt=1, NormalizedWeight=False):

        self.NumberOfSynapses = NumberOfSynapses

        self.dt = dt
        self.t = t
        self.PeriodCentre = np.ones(NumberOfSynapses).astype(np.float) * Period if Period is not None else 1000 + 100 * (np.random.rand(*NumberOfSynapses) - 0.5)
        self.Period = copy.deepcopy(self.PeriodCentre)
        self.tInPeriod = np.ones(NumberOfSynapses).astype(np.float) * tInPeriod if tInPeriod is not None else np.random.rand(*NumberOfSynapses) * self.Period

        self.PeriodVar = np.ones(NumberOfSynapses).astype(np.float) * PeriodVar if PeriodVar is not None else np.ones(NumberOfSynapses).astype(np.float) * 0.1
        self.Amp = np.ones(NumberOfSynapses).astype(np.float) * Amp if Amp is not None else np.ones(NumberOfSynapses).astype(np.float) * 0.2

        self.WeightersCentre = np.ones(NumberOfSynapses) * WeightersCentre if WeightersCentre is not None else (np.random.rand(*NumberOfSynapses) - 0.5) * InitAmp  # 0.4
        self.NormalizedWeight = NormalizedWeight
        # print(self.WeightersCentre)
        # print(np.sum(self.WeightersCentre, axis=1))
        if NormalizedWeight:
            self.WeightersCentre /= np.sum(self.WeightersCentre, axis=1)[:, None]
        self.WeightersCentreUpdateRate = np.ones(
            NumberOfSynapses) * WeightersCentreUpdateRate if WeightersCentreUpdateRate is not None else np.ones(
            NumberOfSynapses) * 0.000012

        self.Weighters = self.WeightersCentre + self.Amp * np.sin(self.tInPeriod / self.Period * 2 * np.pi)
        self.WeightersLast = copy.deepcopy(self.Weighters)
        self.WeightersOscilateDecay = np.ones(NumberOfSynapses) * WeightersOscilateDecay if WeightersOscilateDecay is not None else np.ones(NumberOfSynapses)

        self.ModulatorAmount = np.ones(NumberOfSynapses) * ModulatorAmount
        self.ZeroCross = np.ones(NumberOfSynapses, dtype=bool)


    def StepSynapseDynamics(self, dt, t, ModulatorAmount, PreSynActivity=None):

        if dt is None:
            dt = self.dt
        self.t = t
        self.tInPeriod += dt
        # todo Question
        self.Weighters = self.WeightersCentre + self.Amp * np.sin(self.tInPeriod / self.Period * 2 * np.pi)

        self.WeightersCentre += (self.Weighters - self.WeightersCentre) * ModulatorAmount * self.WeightersCentreUpdateRate * dt

        if self.NormalizedWeight:
            self.WeightersCentre /= np.sum(np.abs(self.WeightersCentre), axis=1)[:, None]

        self.ModulatorAmount = np.ones(self.NumberOfSynapses) * ModulatorAmount
        self.Amp *= np.exp(-self.WeightersOscilateDecay * self.ModulatorAmount * dt)
        self.ZeroCross = np.logical_and(np.less(self.WeightersLast, self.WeightersCentre),
                                        np.greater_equal(self.Weighters, self.WeightersCentre))
        self.tInPeriod[self.ZeroCross] = self.tInPeriod[self.ZeroCross] % self.Period[self.ZeroCross]

        self.Period[self.ZeroCross] = np.random.normal(loc=self.PeriodCentre[self.ZeroCross],
                                                       scale=self.PeriodCentre[self.ZeroCross] * 0.1)
        self.WeightersLast = self.Weighters
        return self.Weighters

    def InitRecording(self):
        self.RecordingState = True
        self.Trace = {
                        'Weighters': deque(),
                        'WeightersCentre' : deque(),
                        'ModulatorAmount' : deque(),
                        'Amp' : deque(),
                        'Period' : deque(),
                        'tInPeriod' : deque(),
                        't': deque()
                        }
    def Recording(self, StepNumber, NumberOfLayer):
        # Temp = None
        # for key in self.Trace:
        #     exec("Temp = self.%s" % (key))
        #     self.Trace[key].append(copy.deepcopy(Temp))
        # self.Trace["StepNumber"] = StepNumber
        # self.Trace["NumberOfLayer"] = NumberOfLayer
        self.Trace['Weighters'].append(self.Weighters)
        self.Trace['WeightersCentre'].append(self.WeightersCentre)
        self.Trace['ModulatorAmount'].append(self.ModulatorAmount)
        self.Trace['Amp'].append(self.Amp)
        self.Trace['Period'].append(self.Period)
        self.Trace['tInPeriod'].append(self.tInPeriod)
        self.Trace['t'].append(self.t)

