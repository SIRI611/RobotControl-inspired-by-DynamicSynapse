#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Test.py
# @Time      :2022/12/3 15:20
# @Author    :Siri

import mujoco_py
import os

mj_path = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)

print(sim.data.qpos)


