import logging
import os

import gym
import numpy as np
from gridworld.tasks.task import Task

from wrappers.target_generator import RandomFigure, Figure, target_to_subtasks

logger = logging.getLogger(__file__)
IGLU_ENABLE_LOG = os.environ.get('IGLU_ENABLE_LOG', '')


class MultitaskFormat(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.figure = None

    def reset(self):
        self.figure = Figure(figure=self.env.task.target_grid)
        return super().reset()


class TargetGenerator(gym.Wrapper):
    def __init__(self, env, make_holes=False, make_colors=False, fig_generator=RandomFigure):
        super().__init__(env)
        self.make_holes = make_holes
        self.make_colors = make_colors
        self.figure = fig_generator()

    def reset(self):
        X = []
        self.figure.make_task()
        if isinstance(self.figure, RandomFigure):
            min_block_in_fig = 10
        else:
            min_block_in_fig = 0
        while len(X) <= min_block_in_fig:
            self.figure.make_task()
            relief = self.figure.figure_parametrs['relief']
            X, Y = np.where(relief != 0)
        return super().reset()


class SubtaskGenerator(gym.Wrapper):
    def __init__(self, env, steps_to_task=150):
        super().__init__(env)
        self.relief_map = None
        self.task_generatir = None
        self.current_grid = np.zeros((9, 11, 11))
        self.old_grid = np.zeros((9, 11, 11))
        self.preinited_grid = None
        self.old_preinited_grid = None
        self.prebuilds_percent = 0.5
        self.last_agent_rotation = (0, 0)
        self.last_target = None
        self.color_map = None
        self.steps_to_task = steps_to_task
        self.steps = 0
        self.name = "Nothing"
        self.tasks = dict()
        self.original = None

    def init_relief(self, count_blocks):
        count_blocks = int(count_blocks)
        p = 1 if count_blocks <= 2 else self.prebuilds_percent
        rangex = list(range(0, count_blocks - 1))
        prob = [p] + [(1 - p) / (count_blocks - 2) for _ in range(count_blocks - 2)]
        try:
            prebuilded = np.random.choice(rangex, p=prob)
        except:
            prebuilded = 0
        self.current_grid = np.zeros((9, 11, 11))
        for i in range(prebuilded):
            coord, custom_grid = next(self.generator)
            x, z, y, id = coord
            id = id / abs(id)
            self.current_grid[z + 1, x + 5, y + 5] += id

        self.current_grid[self.current_grid < 0] = 0
        self.current_grid[self.current_grid > 0] = 1

        blocks = np.where(self.current_grid)
        ind = np.lexsort((blocks[0], blocks[2], blocks[1]))
        Zorig, Xorig, Yorig = blocks[0][ind] - 1, blocks[1][ind] - 5, blocks[2][ind] - 5
        ids = [1] * len(Zorig)
        starting_grid = list(zip(Xorig, Zorig, Yorig, ids))
        return starting_grid, prebuilded

    def init_agent(self, task, last_block):
        X, Y = last_block[0], last_block[2]
        Z = last_block[1] + 2
        return X, Z, Y

    def make_new_task(self):
        self.generator = target_to_subtasks(self.env.figure)
        size = int(len(np.where(self.env.figure.figure_parametrs['figure'] != 0)[0]) * 0.6)
        starting_grid, prebuilded = self.init_relief(size)
        try:
            task = next(self.generator)
        except:
            raise Exception(f"""Subtasks are over! 
                            Count of prebuilds:
                            {prebuilded}
                            Count of blocks:
                            {size}
                            """ % self.env.figure.relief.sum())
        coord, custom_grid = task
        if prebuilded != 0:
            X, Z, Y = self.init_agent(coord, starting_grid[-1])
        else:
            X, Y = np.random.randint(-5, 5, 2)
            Z = 0
        initial_position = (X, Z, Y, self.last_agent_rotation[-2], self.last_agent_rotation[-1])
        return starting_grid, initial_position, custom_grid

    def update_field(self, new_block=None, do=1):
        self.old_grid = self.current_grid[:, :, :]
        self.current_grid[new_block] = do
        self.new_blocks.append((*new_block, do))

    def one_round_reset(self, new_block=None, do=1):
        self.last_target = np.where(self.env.task.target_grid != 0)
        self.steps = 0
        try:
            coord, task = next(self.generator)
        except StopIteration:
            return True
        self.update_field(new_block, do=do)
        self.env.task = Task("", task)
        self.agent_win = False
        return False

    def reset(self):
        obs = super().reset()
        self.last_target = np.where(self.env.task.target_grid != 0)
        self.old_grid = self.current_grid[:, :, :]
        self.steps = 0
        self.done_obs = None
        self.current_grid = np.zeros((9, 11, 11))
        self.old_grid = np.zeros((9, 11, 11))
        self.old_preinited_grid = self.preinited_grid
        self.preinited_grid = np.zeros((9, 11, 11))
        self.new_blocks = []
        starting_grid, initial_position, task = self.make_new_task()
        self.env.initialize_world(starting_grid, initial_position)
        if self.old_preinited_grid is None:
            self.old_preinited_grid = self.preinited_grid
        self.env.task = Task("", task)
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.done_obs = obs['grid']
        self.steps += 1
        self.last_agent_rotation = obs['agentPos'][3:]
        if self.steps >= self.steps_to_task:
            done = True
        return obs, reward, done, info
