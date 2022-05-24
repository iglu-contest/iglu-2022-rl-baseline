

from pandas import NA
import pyglet

pyglet.options["headless"] = True
from gridworld.world import World
from gridworld.control import Agent
from gridworld.render import Renderer, setup
from gridworld.task import Task

from gym.spaces import Dict, Box, Discrete
from gym import Env, Wrapper
import numpy as np
import cv2
from copy import copy
from math import fmod
from uuid import uuid4
import cv2
import os
from collections import defaultdict

class Visual(Wrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.c = None
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.w = None
        self.data = defaultdict(list)
        self.logging = False
        self.turned_off = True
        self.glob_step = 0
        self.size = 256
     #   print(self.observation_space.spaces)

        self.observation_space.spaces['obs'] = Box(low=0, high=1, shape=(self.size, self.size, 3), dtype=np.float32)

    def turn_on(self):
        self.turned_off = False

    def set_idx(self, ix, glob_step):
        self.c = ix
        self.glob_step = glob_step
        self.w = cv2.VideoWriter(f'episodes/step{self.glob_step}_ep{self.c}.mp4', self.fourcc, 20, (self.size,self.size))

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # pov = self.env.render()
        # self.w.write(pov)
        pov = self.env.render()[..., :-1]
        obs['obs'] = pov.astype(np.float32) / 255.
        if not self.turned_off:

            if self.logging:
                for key in obs:
                    self.data[key].append(obs[key])
                self.data['reward'].append(reward)
                self.data['done'].append(done)
                self.w.write(pov)
        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        if not self.turned_off:
            if self.logging:
                if not os.path.exists('episodes'):
                    os.makedirs('episodes', exist_ok=True)
                for k in self.data.keys():
                    self.data[k] = np.stack(self.data[k], axis=0)
                np.savez_compressed(f'episodes/step{self.glob_step}_ep{self.c}.npz', **self.data)
                self.data = defaultdict(list)
                self.w.release()
                fname = f'step{self.glob_step}_ep{self.c}'
                os.system(f'ffmpeg -y -hide_banner -loglevel error -i episodes/{fname}.mp4 -vcodec libx264 episodes/{fname}1.mp4 '
                          f'&& mv episodes/{fname}1.mp4 episodes/{fname}.mp4')
                self.w = None
                self.c += 1000
                self.w = cv2.VideoWriter(f'episodes/step{self.glob_step}_ep{self.c}.mp4', self.fourcc, 20, (self.size,self.size))
        obs['obs'] = self.env.render()[..., :-1].astype(np.float32) / 255.

        return obs

    def enable_renderer(self):
        self.env.enable_renderer()
        self.logging = True
