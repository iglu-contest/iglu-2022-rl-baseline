import sys

import numpy as np
from gridworld.env import GridWorld
from gridworld.tasks.task import Task
from sample_factory.algorithms.appo.enjoy_appo import enjoy
from sample_factory.algorithms.appo.model_utils import register_custom_encoder
from sample_factory.algorithms.utils.arguments import parse_args
from sample_factory.envs.env_registry import global_env_registry

from models.models import ResnetEncoderWithTarget
from wrappers.common_wrappers import VectorObservationWrapper, \
    ColorWrapper, JumpAfterPlace
from wrappers.loggers import VideoLogger, Logger
from wrappers.multitask import SubtaskGenerator, TargetGenerator
from wrappers.reward_wrappers import RangetRewardFilledField
from wrappers.target_generator import  RandomFigure


def make_iglu(*args, **kwargs):
    custom_grid = np.ones((9, 11, 11))
    env = GridWorld(render=True, select_and_place=True, discretize=True, max_steps=1000, render_size=(512, 512))
    env.set_task(Task("", custom_grid))
    figure_generator = RandomFigure
    env = TargetGenerator(env, fig_generator=figure_generator)
    env = SubtaskGenerator(env)
    env = VectorObservationWrapper(env)

    env = JumpAfterPlace(env)
    env = ColorWrapper(env)
    env = RangetRewardFilledField(env)
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
                           '--train_dir=../train_dir/0005'], evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
