#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Main.py
# @Time      :2022/11/29 21:54
# @Author    :Siri
import gym
import numpy as np
import Neural_Network as nn
from NN_Layers import InitLayers


if __name__ == "__main__":
    experiment = 'BipedalWalker-v3'
    env = gym.make(experiment)

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    observation = env.reset()

    NN_ARCHITECTURE = [
        {"INPUT_DIM1": 43,"OUTPUT_DIM1": 8},
        {"INPUT_DIM2": 9, "OUTPUT_DIM2": 2},
        {"INPUT_DIM3": 2, "OUTPUT_DIM3": 4},
        {"INPUT_DIM4": 19, "OUTPUT_DIM4": 4},
    ]
    RecordAll = []
    dt = 33
    N_episode = 10
    T = 0
    NeuronSensitivity = np.ones(NN_ARCHITECTURE[0]["OUTPUT_DIM1"])*0.5
    action = np.zeros(num_actions)

    NN = InitLayers(NN_ARCHITECTURE)
    print(NN["Layer1"].Weighters.shape)
    for i in range(N_episode):
        print(i)
        count = 0
        while True:
            count += 1
            print(count)
            T += dt
            env.render()
            RecordOfStep = nn.ForwardPropagation(observation, action, dt, i, RecordAll, NN, NeuronSensitivity)
            action = RecordOfStep["ActionOutput"]
            NeuronSensitivity = nn.NeuronSensitivityUpdate(NeuronSensitivity, RecordOfStep, UpdateRate=00000.1, dt=dt)
            print(NeuronSensitivity)
            observation, reward, done, info = env.step(action)
            nn.NetworkDynamics(NN,RecordOfStep, reward, T, dt, i)
            #print(env.step(action))
            if done:
                env.reset()
                break

    env.close()


