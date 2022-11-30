#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :NN_Layers.py
# @Time      :2022/11/29 16:22
# @Author    :Siri
import numpy as np
import DynamicSynapse
import FitzHughNagumo
import Functions

NN_ARCHITECTURE = [
    {"INPUT_DIM1": 43, "OUTPUT_DIM1": 8},
    {"INPUT_DIM2": 9, "OUTPUT_DIM2": 2},
    {"INPUT_DIM3": 2, "OUTPUT_DIM3": 4},
    {"INPUT_DIM4": 18, "OUTPUT_DIM4": 4},
]


def InitLayers(NN_ARCHITECTURE):
# init
    NN = dict()
    inputsize1 = NN_ARCHITECTURE[0]["INPUT_DIM1"]
    outputsize1 = NN_ARCHITECTURE[0]["OUTPUT_DIM1"]
    inputsize2 = NN_ARCHITECTURE[1]["INPUT_DIM2"]
    outputsize2 = NN_ARCHITECTURE[1]["OUTPUT_DIM2"]
    inputsize3 = NN_ARCHITECTURE[2]["INPUT_DIM3"]
    outputsize3 = NN_ARCHITECTURE[2]["OUTPUT_DIM3"]
    inputsize4 = NN_ARCHITECTURE[3]["INPUT_DIM4"]
    outputsize4 = NN_ARCHITECTURE[3]["OUTPUT_DIM4"]

# FirstLayer
    NumberOfSynapses1 = [outputsize1, inputsize1]
     # WeightersCentre = np.ones(NumberOfSynapses1) * 0.1 + 0.2 * (np.random.rand(*NumberOfSynapses1) - 0.5)
    ASDA1 = DynamicSynapse.DynamicSynapseArray(NumberOfSynapses1 , Period=20000, tInPeriod=None, PeriodVar=0.1,
                                        Amp=0.2, WeightersCentre = None, WeightersCentreUpdateRate = 0.000012,
                                        WeightersOscilateDecay=0.0000003/100)
    ASDA1.InitRecording()
    NN["Layer1"] = ASDA1


    #SecondLayer
    NumberOfSynapses2 = [outputsize2, inputsize2]
    ADSA2 = DynamicSynapse.DynamicSynapseArray(NumberOfSynapses2, Period=20000, tInPeriod=None, PeriodVar=0.1, \
                                    Amp=0.2, WeightersCentre=None, WeightersCentreUpdateRate=0.000012,
                                    WeightersOscilateDecay=0.0000003 / 100)
    ADSA2.InitRecording()
    NN["Layer2"] = ADSA2


# ThirdLayer/FH
    FN = FitzHughNagumo.FHNN(NumberOfNeurons=inputsize3, scale=0.02)
    FN.InitRecording()
    NN["Layer3"] = FN


# ForthLayer
    WeightersCentre=np.array([[0,0,-0.5,0,0,0,0,0,-0.2,-0.2,0,0,0,0,0,0,0,0,0],
                               [0.5,0,0,0,0,0,0,0,0,0,-0.2,-0.2,0,0,0,0,0,0,0],
                               [0,0,0,-0.5,0,0,0,0,0,0,0,0,0,-0.2,-0.2,0,0,0,0],
                               [0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.2,-0.2,0,0]])
    NumberOfSynapses4 = [outputsize4, inputsize4]
    ADSA4 = DynamicSynapse.DynamicSynapseArray(NumberOfSynapses4, Period=20000, tInPeriod=None, PeriodVar=0.1, \
                                    Amp=0.2, WeightersCentre=WeightersCentre, WeightersCentreUpdateRate=0.000012,
                                    WeightersOscilateDecay=0.0000003 / 100)
    ADSA4.InitRecording()
    NN["Layer4"] = ADSA4

    return NN

