import logging
import os

import gym
import numpy as np

from gridworld.task import Task
from wrappers.target_generator import RandomFigure, DatasetFigure, target_to_subtasks

# from iglu.tasks import RandomTasks

logger = logging.getLogger(__file__)
IGLU_ENABLE_LOG = os.environ.get('IGLU_ENABLE_LOG', '')


class Multitask(gym.Wrapper):
    def __init__(self, env, make_holes=False, make_colors=False, steps_to_task=300, fig_generator=RandomFigure):
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
        self.make_holes = make_holes
        self.make_colors = make_colors
        self.steps = 0
        self.name = "Nothing"
        self.tasks = dict()
        self.original = None
        self.fig_generator = fig_generator()

    def init_relief(self, relief, hole_relief, color_relief, count_blocks):
        """
        Generate relief at the begining of episode.
        Init position of blocks and agent postion.

        """
        p = 1 if count_blocks <= 2 else self.prebuilds_percent  # вероятность старта с начала
        rangex = list(range(0, count_blocks - 1))
        prob = [p] + [(1 - p) / (count_blocks - 2) for i in range(count_blocks - 2)]
        try:
            prebuilded = np.random.choice(rangex, p=prob)
        except:
            prebuilded = 0
        starting_grid = []
        for i in range(prebuilded):
            coord, _ = next(self.generator)
            #  print("COORD -> -> ->", coord)
            if coord[-1] > 0:
                starting_grid.append(coord)
                if (color_relief is None) or color_relief[coord[1] + 1, coord[0] + 5, coord[2] + 5] == 0:
                    self.current_grid[coord[1] + 1, coord[0] + 5, coord[2] + 5] = 1
                else:
                    #  print(color_relief[coord[1] + 1, coord[0] + 5, coord[2] + 5])
                    self.current_grid[coord[1] + 1, coord[0] + 5, coord[2] + 5] = color_relief[
                        coord[1] + 1, coord[0] + 5, coord[2] + 5]
                prev_act_is_add = True
            else:
                if prev_act_is_add:
                    x, z, y, id = starting_grid[-1]
                    Z = z + 1
                    prev_act_is_add = False
                coord = starting_grid.pop(-Z)
                self.current_grid[coord[1] + 1, coord[0] + 5, coord[2] + 5] = 0
                Z -= 1
        self.preinited_grid[:, :, :] = self.current_grid
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
        X = []
        self.fig_generator.make_task()
        if self.fig_generator.use_color:
            min_block_in_fig = 0
        else:
            min_block_in_fig = 9
        while len(X) <= min_block_in_fig:
            self.fig_generator.make_task()
            relief = self.fig_generator.relief
            X, Y = np.where(relief != 0)

        if self.make_holes:
            hole_relief = self.fig_generator.hole_relief
        else:
            hole_relief = None

        if self.make_colors and self.fig_generator.use_color:
            color_relief = self.fig_generator.color
        else:
            color_relief = None
        if isinstance(self.fig_generator, DatasetFigure):
            if self.name not in self.tasks:
                self.tasks[self.name] = 1
            else:
                self.tasks[self.name] += 1
            self.original = self.fig_generator.original
            self.name = self.fig_generator.name
            self.f1_onstart = self.fig_generator.f1_onstart
            self.rp = self.fig_generator.is_right_predicted
            self.color_map = color_relief
            self.modified = self.fig_generator.is_modified
            self.chat = self.fig_generator.chat
        self.relief_map = relief
        self.hole_map = hole_relief
        self.generator = target_to_subtasks(relief, hole_relief, color_relief)
        starting_grid, prebuilded = self.init_relief(relief, hole_relief, color_relief, len(X))
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
        #   print(f"!!!!! >>>>>> {np.sum(task)}")
        except StopIteration:
            return True

        self.update_field(new_block, do=do)

        self.env.task = Task("", task)
        self.agent_win = False
        return False

    def reset(self):
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
        return super().reset()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if done:
            raise Exception("catch done")
        self.done_obs = obs['grid']
        self.steps += 1
        self.last_agent_rotation = obs['agentPos'][3:]
        return obs, reward, done, info
