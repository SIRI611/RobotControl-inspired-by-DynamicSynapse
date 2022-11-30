#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :ChoosePath.py
# @Time      :2022/11/28 20:01
# @Author    :Siri
import os.path
import platform

def ChooseRecordingPath(experiment, TimeOfRecording):
    if platform.node() == "LAPTOP-68CSC593":
        path = "F:/NeurobotsTrain/Recording/OpenaiGym/" + experiment + TimeOfRecording + "/"
    else :
        path = ''
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def ChooseResultPath(experiment, TimeOfRecording):
    if platform.node() == "LAPTOP-68CSC593":
        path = "F:/NeurobotsTrain/Recording/OpenaiGym/SimulateResult/" + experiment + TimeOfRecording + "/"
    else:
        path = ''
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def ChooseCodePath(experiment, TimeOfRecording):
    if platform.node() == "LAPTOP-68CSC593":
        path = "F:/NeurobotsTrain/Recording/OpenaiGym/" + experiment + TimeOfRecording + "Src/"
    else:
        path = ''
    if not os.path.exists(path):
        os.mkdir(path)
    return path