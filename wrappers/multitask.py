import logging
import os

import gym
import numpy as np

from gridworld.task import Task
from wrappers.target_generator import RandomFigure, DatasetFigure, target_to_subtasks

# from iglu.tasks import RandomTasks

logger = logging.getLogger(__file__)
IGLU_ENABLE_LOG = os.environ.get('IGLU_ENABLE_LOG', '')

class TargetGenerator(gym.Wrapper):
    def __init__(self, env, make_holes=False, make_colors=False,fig_generator=RandomFigure):
        super().__init__(env)
      #  self.fig_generator = fig_generator()
        self.make_holes = make_holes
        self.make_colors = make_colors
        self.figure = fig_generator()

    def reset(self):
        X = []
        self.figure.make_task()
        if self.figure.use_color:
            min_block_in_fig = 0
        else:
            min_block_in_fig = 9
        while len(X) <= min_block_in_fig:
            self.figure.make_task()
            relief = self.figure.figure_parametrs['relief']
            X, Y = np.where(relief != 0)
        print("make figure")

      #  self.env.task = Task("", self.figure.figure_parametrs['figure'])
        return super().reset()



class SubtaskGenerator(gym.Wrapper):
    def __init__(self, env,  steps_to_task=300):
        super().__init__(env)
        self.relief_map = None
        self.task_generatir = None
        self.current_grid = np.zeros((9, 11, 11))
        self.old_grid = np.zeros((9, 11, 11))
        self.preinited_grid = None
        self.old_preinited_grid = None
        self.prebuilds_percent = 0.3
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
        p = 1 if count_blocks <= 2 else self.prebuilds_percent  # вероятность старта с начала
        rangex = list(range(0, count_blocks - 1))
        prob = [p] + [(1 - p) / (count_blocks - 2) for i in range(count_blocks - 2)]
        try:
            prebuilded = np.random.choice(rangex, p=prob)
        except:
            prebuilded = 0
        blocks = np.where(self.env.figure.figure_parametrs['figure']!=0)
        Z,X,Y = (blocks[0][:prebuilded]-1,
                         blocks[1][:prebuilded]-5,
                         blocks[2][:prebuilded]-5)
        idx = np.ones_like(X)
        starting_grid = list(zip(X,Z,Y,idx))

        self.current_grid = np.zeros((9,11,11))
        self.current_grid[Z+1,X+5,Y+5] = 1
        return starting_grid, prebuilded

    def init_agent(self, task, last_block):
        if task[-1] < 0:
            if task[1] > 0:
                add = [(1, 0), (0, 1), (1, 1)]
                ind = np.random.randint(0, 3)
                X, Y = last_block[0] + add[ind][0], last_block[2] + add[ind][1]
                Z = -1
            else:
                X, Y = last_block[0], last_block[2]
                Z = last_block[1] + 1
        X, Y = last_block[0], last_block[2]
        Z = last_block[1] + 1
        return X, Z, Y

    def make_new_task(self):
        self.generator = target_to_subtasks(self.env.figure)
        size = self.env.figure.figure_parametrs['relief'].sum()
        starting_grid, prebuilded = self.init_relief( size)
        try:
            task = next(self.generator)
        except:
            raise Exception("Subtasks are over! Relief map sum:  %d"%self.relief_map.sum())
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
        # print("UPD -relief", new_block, do)
        self.current_grid[new_block] = do
        self.new_blocks.append((*new_block, do))

    def one_round_reset(self, new_block=None, do=1):  # dowin
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
       # if done:
        #    raise Exception("catch done")
        self.done_obs = obs['grid']
        self.steps += 1
        self.last_agent_rotation = obs['agentPos'][3:]
        return obs, reward, done, info
