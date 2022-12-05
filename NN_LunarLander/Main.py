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
    experiment = 'LunarLander-v2'
    env = gym.make(experiment)

    # num_states = env.observation_space.shape[0]
    num_actions = env.action_space
    print(num_actions)
    observation = env.reset()

    NN_ARCHITECTURE = [
        {"INPUT_DIM1": 16,"OUTPUT_DIM1": 8},
        {"INPUT_DIM2": 9, "OUTPUT_DIM2": 4},
        {"INPUT_DIM3": 11, "OUTPUT_DIM3": 4},
        {"INPUT_DIM4": 4, "OUTPUT_DIM4": 1}
    ]
    RecordAll = []
    dt = 33
    N_episode = 100
    T = 0
    NeuronSensitivity = np.ones(NN_ARCHITECTURE[0]["OUTPUT_DIM1"])*0.5
    action = 1
    NN = InitLayers(NN_ARCHITECTURE)
    # print(NN["Layer1"].Weighters.shape)
    for i in range(N_episode):
        #print(i)
        count = 0
        while True:
            count += 1
            print("EpisodeNumber  ",i)
            print("StepNumber:  ",count)
            T += dt
            env.render()
            RecordOfStep = nn.ForwardPropagation(observation, action, dt, count, RecordAll, NN, NeuronSensitivity, i)
            action = RecordOfStep["ActionOutput"]
            print(action)
            NeuronSensitivity = nn.NeuronSensitivityUpdate(NeuronSensitivity, RecordOfStep, UpdateRate=00000.1, dt=dt)
            #print(NeuronSensitivity)
            if action <= 0:
                observation, reward, done, info = env.step(0)
            if action > 0 and action <= 0.3:
                observation, reward, done, info = env.step(1)
            if action > 0.3 and action <= 0.7 :
                observation, reward, done, info = env.step(2)
            if action > 0.7 and action <= 1:
                observation, reward, done, info = env.step(3)

            nn.NetworkDynamics(NN, RecordOfStep, reward, T, dt, i)
            #print(env.step(action))
            if done:
                env.reset()
                break
    # print(NN["Layer1"].Trace)
    env.close()


