import logging
import os
from collections import OrderedDict
from typing import Generator

import gym
import numpy as np

from wrappers.target_generator import RandomFigure

logger = logging.getLogger(__file__)
IGLU_ENABLE_LOG = os.environ.get('IGLU_ENABLE_LOG', '')


class Wrapper(gym.Wrapper):
    def stack_actions(self):
        if isinstance(self.env, Wrapper):
            return self.env.stack_actions()

    def wrap_observation(self, obs, reward, done, info):
        if hasattr(self.env, 'wrap_observation'):
            return self.env.wrap_observation(obs, reward, done, info)
        else:
            return obs


class ActionsWrapper(Wrapper):
    def wrap_action(self, action) -> Generator:
        raise NotImplementedError

    def stack_actions(self):
        def gen_actions(action):
            for action in self.wrap_action(action):
                wrapped = None
                if hasattr(self.env, 'stack_actions'):
                    wrapped = self.env.stack_actions()
                if wrapped is not None:
                    yield from wrapped(action)
                else:
                    yield action

        return gen_actions

    def step(self, action):
        total_reward = 0
        for a in self.wrap_action(action):
            obs, reward, done, info = super().step(a)
            total_reward += reward
            if done:
                return obs, total_reward, done, info
        return obs, total_reward, done, info


class ObsWrapper(Wrapper):
    def observation(self, obs, reward=None, done=None, info=None):
        raise NotImplementedError

    def wrap_observation(self, obs, reward, done, info):
        new_obs = self.observation(obs, reward, done, info)
        return self.env.wrap_observation(new_obs, reward, done, info)

    def reset(self):
        return self.observation(super().reset())

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info['grid'] = obs['grid']
        info['agentPos'] = obs['agentPos']
       # info['obs'] = obs['pov']
        return self.observation(obs, reward, done, info), reward, done, info


def flat_action_space(action_space):
    if action_space == 'human-level':
        return flat_human_level
    elif action_space == 'flying':
        return flat_flying
    else:
        raise Exception("Acton space not found!")


def no_op():
    return OrderedDict([('attack', 0), ('back', 0), ('camera', np.array([0., 0.])),
                        ('forward', 0), ('hotbar', 0), ('jump', 0), ('left', 0), ('right', 0),
                        ('use', 0)])


def flat_human_level(env, camera_delta=5):
    binary = ['attack', 'forward', 'back', 'left', 'right', 'jump']
    discretes = [no_op()]
    for op in binary:
        dummy = no_op()
        dummy[op] = 1
        discretes.append(dummy)
    camera_x = no_op()
    camera_x['camera'][0] = camera_delta
    discretes.append(camera_x)
    camera_x = no_op()
    camera_x['camera'][0] = -camera_delta
    discretes.append(camera_x)
    camera_y = no_op()
    camera_y['camera'][1] = camera_delta
    discretes.append(camera_y)
    camera_y = no_op()
    camera_y['camera'][1] = -camera_delta
    discretes.append(camera_y)
    for i in range(6):
        dummy = no_op()
        dummy['hotbar'] = i + 1
        discretes.append(dummy)
    discretes.append(no_op())
    return discretes

def no_op_f():
    return OrderedDict([('movement',np.array([0., 0., 0.])),
                         ('camera', np.array([0., 0.])),
                        ('inventory', 0),
                        ('placement', 0)])

def flat_flying(env, camera_delta=5, step_delta = 1):
    
    discretes = [no_op_f()]
    
    ###### blocks
    for color in range(0,7):
        dummy = no_op_f()
        dummy['inventory'] = color
        discretes.append(dummy)
              
    ###### placement
    for move in range(0,3):
        dummy = no_op_f()
        dummy['placement'] = move
        discretes.append(dummy)
                        
    ###### camera                    
    camera_x = no_op_f()
    camera_x['camera'][0] = camera_delta
    discretes.append(camera_x)
                        
    camera_x = no_op_f()
    camera_x['camera'][0] = -camera_delta
    discretes.append(camera_x)
                        
    camera_y = no_op_f()
    camera_y['camera'][1] = camera_delta
    discretes.append(camera_y)
                        
    camera_y = no_op_f()
    camera_y['camera'][1] = -camera_delta
    discretes.append(camera_y)
    
    #### movement
    move_x = no_op_f()
    move_x['movement'][0] = step_delta
    discretes.append(move_x)
                        
    move_y = no_op_f()
    move_y['movement'][0] = step_delta
    discretes.append(move_y)
                        
    move_z = no_op_f()
    move_z['movement'][0] = step_delta
    discretes.append(move_z)
                        
    move_x = no_op_f()
    move_x['movement'][0] = -step_delta
    discretes.append(move_x)
                        
    move_y = no_op_f()
    move_y['movement'][0] = -step_delta
    discretes.append(move_y)
                        
    move_z = no_op_f()
    move_z['movement'][0] = -step_delta
    discretes.append(move_z)
                        
    return discretes


class Discretization(ActionsWrapper):
    def __init__(self, env, flatten = 'flying'):
        super().__init__(env)
        camera_delta = 5
        step_delta = 1
        self.discretes = flat_flying(env, camera_delta, step_delta)
        self.action_space = gym.spaces.Discrete(len(self.discretes))
        self.old_action_space = env.action_space
        self.last_action = None

    def wrap_action(self, action=None, raw_action=None):
        if action is not None:
            action = self.discretes[action]
        elif raw_action is not None:
            action = raw_action
        yield action


class JumpAfterPlace(ActionsWrapper):
    def __init__(self, env):
        min_inventory_value = 5
        max_inventory_value = 12
        self.act_space = (min_inventory_value, max_inventory_value)
        super().__init__(env)

    def wrap_action(self, action=None):
        if (action > self.act_space[0]) and (action < self.act_space[1]) > 0:
            yield action
            yield 5
            yield 5
            
           # yield 5
        else:
            yield action


class ColorWrapper(ActionsWrapper):
    def __init__(self, env):
        super().__init__(env)
        min_inventory_value = 5
        max_inventory_value = 12
        self.color_space = (min_inventory_value, max_inventory_value)

    def wrap_action(self, action=None):
        tcolor = np.sum(self.env.task.target_grid)
        if (action > self.color_space[0]) and (action < self.color_space[1]) and tcolor > 0:
            if isinstance(self.env.figure, RandomFigure):
                action = int(self.color_space[0] + np.random.randint(1, 6))
            else:
                action = int(self.color_space[0] + tcolor)
        yield action


class VisualObservationWrapper(ObsWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.colums = None
        if 'pov' in self.env.observation_space.keys():
            self.observation_space = gym.spaces.Dict({
                'compass':  gym.spaces.Box(low=-180, high=180, shape=(1,)),
                'inventory': gym.spaces.Box(low=0.0, high=20.0, shape=(6,)),
                'target_grid': gym.spaces.Box(low=0.0, high=6.0, shape=(9, 11, 11)),
                'grid': gym.spaces.Box(low=0.0, high=6.0, shape=(9, 11, 11)),
                'obs': gym.spaces.Box(low=0, high=1, shape=(self.env.render_size[0], self.env.render_size[0], 3),
                                      dtype=np.float32),
                'agentPos': gym.spaces.Box(low=-5000.0, high=5000.0, shape=(5,))
            })
        else:
            raise Exception("It is visual wprapper! Obs not found!")

    def observation(self, obs, reward=None, done=None, info=None):
        if IGLU_ENABLE_LOG == '1':
            self.check_component(
                obs['agentPos'], 'agentPos', self.observation_space['agentPos'].low,
                self.observation_space['agentPos'].high
            )
            self.check_component(
                obs['inventory'], 'inventory', self.observation_space['inventory'].low,
                self.observation_space['inventory'].high
            )
            self.check_component(
                obs['grid'], 'grid', self.observation_space['grid'].low,
                self.observation_space['grid'].high
            )
        if info is not None:
            if 'target_grid' in info:
                target_grid = self.env.task.target_grid
            else:
                # logger.error(f'info: {info}')
                if hasattr(self.unwrapped, 'should_reset'):
                    self.unwrapped.should_reset(True)
                target_grid = self.env.task.target_grid
        else:
            target_grid = self.env.task.target_grid

        if 'pov' in self.env.observation_space.keys():
            return {
                'agentPos': obs['agentPos'],
                'compass': obs['compass'],
                'inventory': obs['inventory'],
                'target_grid': target_grid,
                'grid': obs['grid'],
                'obs': obs['pov']
            }
        else:
            raise Exception("It is visual wprapper! Obs not found!")

    def check_component(self, arr, name, low, hi):
        if (arr < low).any():
            logger.info(f'{name} is below level {low}:')
            logger.info((arr < low).nonzero())
            logger.info(arr[arr < low])
        if (arr > hi).any():
            logger.info(f'{name} is above level {hi}:')
            logger.info((arr > hi).nonzero())
            logger.info(arr[arr > hi])
