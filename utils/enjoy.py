import sys

import numpy as np
from sample_factory.algorithms.appo.enjoy_appo import enjoy
from sample_factory.algorithms.appo.model_utils import register_custom_encoder
from sample_factory.algorithms.utils.arguments import parse_args
from sample_factory.envs.env_registry import global_env_registry

from gridworld.env import GridWorld
from models.models import ResnetEncoderWithTarget
from visual import Visual
from wrappers.common_wrappers import VectorObservationWrapper, \
    Discretization, flat_action_space, ColorWrapper, JumpAfterPlace
from wrappers.loggers import VideoLogger
from wrappers.multitask import SubtaskGenerator, TargetGenerator
from wrappers.reward_wrappers import RangetRewardFilledField
from wrappers.target_generator import DatasetFigure


def make_iglu(*args, **kwargs):
    custom_grid = np.ones((9, 11, 11))
    env = GridWorld(custom_grid, render=True, select_and_place=True, max_steps=2050)
    env = Visual(env)  #

    figure_generator = DatasetFigure
    env = TargetGenerator(env, fig_generator=figure_generator)
    env = SubtaskGenerator(env)
    env = VectorObservationWrapper(env)
    env = Discretization(env, flat_action_space('human-level'))
    env = JumpAfterPlace(env)
    env = ColorWrapper(env)
    env = RangetRewardFilledField(env)
    env = VideoLogger(env)
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
