#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :keyBoardControl.py
# @Time      :2022/11/28 21:16
# @Author    :Siri
from pyglet.window import Window

class KeyboardControl:
    def __init__(self, human_reward=0, human_wants_restart=False, human_sets_pause=False,
                 human_stop=False, human_render=False):
        self.human_reward = human_reward
        self.human_wants_restart = human_wants_restart
        self.human_sets_pause = human_sets_pause
        self.human_stop = human_stop
        self.human_render = human_render

    def key_press(self, pressed_key, mod):
        if pressed_key == 0xff0d:
            self.human_wants_restart = True
        if pressed_key == ord('p'):
            self.human_sets_pause = not self.human_sets_pause
            if self.human_sets_pause:
                print("Paused, if want to continue, please press 'p'")
            else:
                print("Continuing, if want to pause, please press 'p'")
        if pressed_key == ord('s'):
            self.human_stop = True
        if pressed_key == ord('r'):
            self.human_render = True
        a = int(pressed_key - ord('0'))
        if a <= 0 or a >= 9:
            return
        self.human_reward = 2 ** (a - 5)


    def key_release(self, pressed_key, mod):
        global human_reward, human_render
        if pressed_key == ord('r'):
            self.human_render = False
        a = int(pressed_key - ord('0'))
        if a <= 0 or a >= 9:
            return
        if human_reward == 2 ** (a - 5):
            self.human_reward = 0
