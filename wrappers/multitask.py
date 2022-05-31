import logging
import os

import gym
import numpy as np

from gridworld.task import Task
from wrappers.target_generator import RandomFigure, Figure, target_to_subtasks

# from iglu.tasks import RandomTasks

logger = logging.getLogger(__file__)
IGLU_ENABLE_LOG = os.environ.get('IGLU_ENABLE_LOG', '')

class MultitaskFormat(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        self.figure = Figure(figure = self.env.task.target_grid)
        return super().reset()


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
            min_block_in_fig = 10
        else:
            min_block_in_fig = 10
        while len(X) <= min_block_in_fig:
            self.figure.make_task()
            relief = self.figure.figure_parametrs['relief']
            X, Y = np.where(relief != 0)
        return super().reset()

class SubtaskGenerator(gym.Wrapper):
    def __init__(self, env,  steps_to_task=150):
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
        p = 1 if count_blocks <= 2 else self.prebuilds_percent  # вероятность старта с начала
        rangex = list(range(0, count_blocks - 1))
        prob = [p] + [(1 - p) / (count_blocks - 2) for i in range(count_blocks - 2)]
        try:
            prebuilded = np.random.choice(rangex, p=prob)
        except:
            prebuilded = 0
        print("FULL FIGURE")
        print(self.env.figure.figure_parametrs['figure'].sum(axis = 0))
        figure = self.env.figure.figure_parametrs['figure'].copy()
        blocks = np.where(figure)
        ind = np.lexsort((blocks[0], blocks[2], blocks[1]))
        Zorig, Xorig, Yorig = blocks[0][ind], blocks[1][ind], blocks[2][ind]
        print(Zorig, Xorig, Yorig)
        Z,X,Y= (Zorig[:prebuilded]-1,
                        Xorig[:prebuilded]-5,
                         Yorig[:prebuilded]-5)
        idx = np.ones_like(X)
        starting_grid = list(zip(X,Z,Y,idx))
        self.current_grid = np.zeros((9,11,11))
        self.current_grid[Z+1,X+5,Y+5] = 1
        print("PREBUILDED!")
        print(self.current_grid.sum(axis = 0))
        return starting_grid, prebuilded, (Zorig, Xorig, Yorig)

    def init_agent(self, task, last_block):
        X, Y = last_block[0], last_block[2]
        Z = last_block[1] + 2
        return X, Z, Y

    def make_new_task(self):
        size = int(self.env.figure.figure_parametrs['relief'].sum()*0.6)
        starting_grid, prebuilded, sorted_blocks_coord = self.init_relief(size)
        Z, X, Y = (sorted_blocks_coord[0][prebuilded:],
                   sorted_blocks_coord[1][prebuilded:],
                   sorted_blocks_coord[2][prebuilded:])
        remains = np.zeros_like(self.env.figure.figure_parametrs['figure'])
        remains[Z,X,Y]=1
        print("LOST!")
        print(remains.sum(axis = 0))
        self.env.figure.to_multitask_format(remains)
        self.env.figure.simplify()
      #  print(self.env.figure.figure_parametrs['figure'].sum(axis=0))
        self.generator = target_to_subtasks(self.env.figure)

        try:
            task = next(self.generator)
        except:
            raise Exception(f"""Subtasks are over! 
                            Relief map sum:  %d
                            Remains: 
                            {remains.sum(axis = 0)} 
                            Blocks:
                            {sorted_blocks_coord}
                            Count of prebuilds:
                            {prebuilded}
                            """%self.env.figure.relief.sum())
        coord, custom_grid = task
        print("TASK")
        print(custom_grid.sum(axis = 0))
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
        if self.steps >= self.steps_to_task:
            done = True
        return obs, reward, done, info
