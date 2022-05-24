import random
import numpy as np
from gym.spaces import Box
from gym.wrappers import TimeLimit
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper

from gridworld.task import Task
from gridworld.env import GridWorld
from wrappers.common_wrappers import  VectorObservationWrapper,  \
    Discretization,  flat_action_space
from wrappers.reward_wrappers import RangetRewardFilledField, Closeness
from wrappers.loggers import SuccessRateWrapper, SuccessRateFullFigure
from wrappers.multitask import Multitask
import gym


class AutoResetWrapper(gym.Wrapper):
    def step(self, action):
        observations, rewards, dones, infos = self.env.step(action)
        if all(dones):
            observations = self.env.reset()
        return observations, rewards, dones, infos


class FakeObsWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = self.env.observation_space
        self.observation_space['obs'] = Box(0.0, 1.0, shape=(1,))

    def observation(self, observation):
        observation['obs'] = np.array([0.0])
        return observation


def make_iglu(*args, **kwargs):

    custom_grid = np.ones((9,11,11))
    env = GridWorld(custom_grid, render=False, select_and_place=True, max_steps=1000)
    env = Multitask(env, make_holes=True)
    env = VectorObservationWrapper(env)
    env = Discretization(env, flat_action_space('human-level'))
    env = RangetRewardFilledField(env)
    env = Closeness(env)
    #env = SuccessRateWrapper(env)
    env = SuccessRateFullFigure(env)
    env = MultiAgentWrapper(env)
    env = AutoResetWrapper(env)

    return env

