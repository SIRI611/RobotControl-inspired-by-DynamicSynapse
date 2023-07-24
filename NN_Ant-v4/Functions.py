#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Functions.py
# @Time      :2022/12/2 21:27
# @Author    :Siri
import numpy as np
import argparse
def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softMax(AList):
    posibility = AList / (np.sum(AList))
    roll = np.random.rand()
    accum = 0
    for i1 in range(len(posibility)):
        if roll > accum and roll < accum + posibility[i1]:
            return i1
        accum += posibility[i1]

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    run_code = 0
