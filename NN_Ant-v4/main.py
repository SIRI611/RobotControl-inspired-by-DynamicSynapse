#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Main.py
# @Time      :2022/12/2 21:40
# @Author    :Siri
import gym
import mujoco_py
import numpy as np
import Neural_Network as nn
from NN_Layers import InitLayers


if __name__ == "__main__":
    experiment = 'Ant-v4'
    env = gym.make(experiment)
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    observation = env.reset()[0]
    ObservationNorm = observation

    NN_ARCHITECTURE = [
        {"INPUT_DIM1": 52,"OUTPUT_DIM1": 16},
        {"INPUT_DIM2": 17, "OUTPUT_DIM2": 4},
        {"INPUT_DIM3": 4, "OUTPUT_DIM3": 8},
        {"INPUT_DIM4": 25, "OUTPUT_DIM4": 8},
    ]
    RecordAll = []
    dt = 33
    N_episode = 10
    T = 0
    NeuronSensitivity = np.ones(NN_ARCHITECTURE[0]["OUTPUT_DIM1"])*0.5
    action = np.zeros(num_actions)

    NN = InitLayers(NN_ARCHITECTURE)
    # print(NN["Layer1"].Weighters.shape)
    for i in range(N_episode):
        #print(i)
        count = 0
        while True:
            count += 1
            # print(count)
            T += dt
            env.render()
            RecordOfStep = nn.ForwardPropagation(ObservationNorm, action, dt, count, RecordAll, NN, NeuronSensitivity, i)
            action = RecordOfStep["ActionOutput"]
            NeuronSensitivity = nn.NeuronSensitivityUpdate(NeuronSensitivity, RecordOfStep, UpdateRate=00000.1, dt=dt)
            #print(NeuronSensitivity)
            # print(env.step(action))
            observation, reward, done, _, info = env.step(action)
            print(observation.shape)
            ObservationNorm = observation
            nn.NetworkDynamics(NN,RecordOfStep, reward, T, dt, i)
            #print(env.step(action))
            if done:
                env.reset()
                break
    # print(NN["Layer1"].Trace)
    env.close()


