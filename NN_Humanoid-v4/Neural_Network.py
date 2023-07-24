#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Neural_Network.py
# @Time      :2022/12/2 21:27
# @Author    :Siri
import numpy as np
import DynamicSynapse
import FitzHughNagumo
import Functions
import copy
import gym
from collections import deque
from NN_Layers import InitLayers
import argparse

def NeuronSensitivityUpdate(NeuronSensitivity, Record, dt, UpdateRate=0.000001):
    # NeuronSensitivity += ((0.3 - np.abs(0 - Record["OutputLayer1"])) * UpdateRate * dt) * (
    #             2 - np.log10(NeuronSensitivity)) * (np.log10(NeuronSensitivity) - (-2)) / 4
    NeuronSensitivity += ((0.3 - np.abs(0 - Record["OutputLayer1"])) * UpdateRate * dt)
    return NeuronSensitivity

def PreProcess(Observation):
    #initial factors
    # factors=np.array([1/2*np.pi,5,2,2,1,
    #          0.5,1,0.3,1,1,
    #          0.5,1,0.25,1,2,
    #          1,1,1,1,1,
    #          1,1,1,1])
    NormalizeOb = Observation[0:45]
    # print(NormalizeOb)
    # NormalizeOb = Observation
    co = Functions.relu(NormalizeOb)
    re = Functions.relu(-NormalizeOb)
    LeftObservation = Observation[45:]
    Combination = np.hstack((co,re,LeftObservation,1))
    print(Combination)
    return Functions.tanh(Combination)

def MergeInputs(Observation, Action):
    Action_norm = Functions.tanh(Action)
    return np.hstack((PreProcess(Observation),Action_norm))


def ForwardPropagation(Observation, action, dt, NumOfStep, RecordAll, NNList, NeuronSensitivity, Episode):

    InputOfLayer1 = MergeInputs(Observation, action)
    OutputOfLayer1 = Functions.relu(np.tanh(np.dot(NNList["Layer1"].Weighters, InputOfLayer1)*NeuronSensitivity))

    InputOfLayer2 = np.hstack((OutputOfLayer1,1))
    OutputOfLayer2 = np.dot(NNList["Layer2"].Weighters, InputOfLayer2)

    InputOfLayer3 = Functions.tanh(OutputOfLayer2)
    OutputOfLayer3 = np.array((NNList["Layer3"].Vn, NNList["Layer3"].Wn)).ravel()

    InputOfLayer4 = np.hstack((OutputOfLayer3, np.hstack((Observation[5:13],Observation[19:27])),1))
    OutputOfLayer4 = np.dot(NNList["Layer4"].Weighters, InputOfLayer4)

    RecordOfInput = dict()
    Record = dict()
    Record["EpisodeNumber"] = Episode
    Record["StepNumber"] = NumOfStep
    Record["InputOfLayer1"] = InputOfLayer1
    Record["InputOfLayer2"] = InputOfLayer2
    Record["InputOfLayer3"] = InputOfLayer3
    Record["InputOfLayer4"] = InputOfLayer4

    Record["OutputLayer1"] = OutputOfLayer1
    Record["OutputLayer2"] = OutputOfLayer2
    Record["OutputLayer3"] = OutputOfLayer3
    Record["OutputLayer4"] = OutputOfLayer4
    Record["ActionOutput"] = Functions.tanh(OutputOfLayer4)


    RecordAll.append(Record)

    return Record

def NetworkDynamics(NN, Record, reward, t, dt, StepNumber):

    NN["Layer1"].Weighters = NN["Layer1"].StepSynapseDynamics(dt, t, reward, PreSynActivity=None)
    NN["Layer1"].Recording(StepNumber,1)

    NN["Layer2"].Weighters = NN["Layer2"].StepSynapseDynamics(dt, t, reward, PreSynActivity=None)
    NN["Layer2"].Recording(StepNumber,2)

    NN["Layer3"].StepDynamics(dt, Record["InputOfLayer3"])
    NN["Layer3"].Update()
    NN["Layer3"].Recording(StepNumber,3)

    NN["Layer4"].Weighters = NN["Layer4"].StepSynapseDynamics(dt, t, reward, PreSynActivity=None)
    NN["Layer4"].Recording(StepNumber,4)

if __name__ == "__main__":
    run_code = 0
