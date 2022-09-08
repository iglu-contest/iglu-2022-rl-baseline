import sys

import numpy as np
from gridworld.env import GridWorld
from gridworld.tasks.task import Task
from sample_factory.algorithms.appo.enjoy_appo import enjoy
from sample_factory.algorithms.appo.model_utils import register_custom_encoder
from sample_factory.algorithms.utils.arguments import parse_args
from sample_factory.envs.env_registry import global_env_registry

sys.path.append("./")
#sys.path.append("../")
import os
cwd = os.getcwd()

from models.model import ResnetEncoderWithTarget
from wrappers.common_wrappers import VisualObservationWrapper, \
    ColorWrapper, JumpAfterPlace
from wrappers.loggers import VideoLogger, Logger, R1_score, SuccessRateFullFigure, Statistics,StatisticsLogger
from wrappers.multitask import SubtaskGenerator, TargetGenerator
from wrappers.reward_wrappers import RangetRewardFilledField

from wrappers.target_generator import  RandomFigure, CustomFigure
import gym


def tasks_from_database():
    names = np.load('data/augmented_target_name.npy')
    targets = np.load('data/augmented_targets.npy')
    
    return dict(zip(names, targets))
 
def castom_tasks():
    tasks = dict()   
    
    t1 = np.zeros((9,11,11))
    t1[0, 1:4, 1:4] = 1
    tasks['[0, 1:4, 1:4]'] = t1
    
    t2 = np.zeros((9,11,11))
    t2[0:2, 1:4, 1:4] = 1
    tasks['[0:2, 1:4, 1:4]'] = t2
    
    t3 = np.zeros((9,11,11))
    t3[0:5, 4, 4] = 1
    t3[1, 4, 4] = 0
    t3[3, 4, 4] = 0
    t3[0, 8, 7] = 1
    tasks['[0:7, 4, 4]'] = t3
    
    t4 = np.zeros((9,11,11))
    t4[0, 4:8, 4:8] = 1
    tasks['[0, 4:8, 4:8]'] = t4
    
    t5 = np.zeros((9,11,11))
    t5[0:3, 8:10, 8:10] = 1
    tasks['[0:3, 8:10, 8:10]'] = t5
    
    return tasks

def make_iglu(*args, **kwargs):
    custom_grid = np.ones((9, 11, 11))
    render = True
   # env = GridWorld(render=True, select_and_place=True, discretize=True, max_steps=1000)
    env = gym.make('IGLUGridworld-v0' if render else 'IGLUGridworldVector-v0',
                    action_space='walking',
                    size_reward=False, max_steps=500, vector_state=True)
    env.set_task(Task("", custom_grid))
    
    figure_generator = CustomFigure
    figure_generator.row_figure[0, 1:4, 1:4] = 1
    figure_generator.generator_name = '[0, 1:4, 1:4]'
    
    tasks = tasks_from_database()
    env = TargetGenerator(env, fig_generator=figure_generator,  tasks = tasks)
    env = SubtaskGenerator(env)
    env = VisualObservationWrapper(env)

    env = JumpAfterPlace(env)
    env = ColorWrapper(env)
    env = RangetRewardFilledField(env)
    
    # Loggers
    env = Statistics(env)
    env = R1_score(env)
    env = SuccessRateFullFigure(env)
    env = StatisticsLogger(env, st_name = "custom_step_by_step_tasks.csv")
    env = VideoLogger(env)
    env = Logger(env)
    return env


def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix='IGLUSilentBuilder-v0',
        make_env_func=make_iglu,
    )
    register_custom_encoder('custom_env_encoder', ResnetEncoderWithTarget)


def main():
    """Script entry point."""
    register_custom_components()
    cfg = parse_args(argv=['--algo=APPO', '--env=IGLUSilentBuilder-v0', '--experiment=TreeChopBaseline-iglu',
                           '--experiments_root=force_envs_single_thread=False;num_envs_per_worker=1;num_workers=10',
                           '--train_dir=./train_dir/0001'], evaluation=True)
    
    status = enjoy(cfg , 1e7)
    return status


if __name__ == '__main__':
    sys.exit(main())
