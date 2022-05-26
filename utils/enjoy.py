
import numpy as np
from sample_factory.envs.env_registry import global_env_registry
from sample_factory.algorithms.appo.model_utils import register_custom_encoder
from sample_factory.algorithms.appo.enjoy_appo import enjoy
from sample_factory.algorithms.utils.arguments import parse_args
import sys

from wrappers.loggers import  VideoLogger
from sample_factory.algorithms.utils.arguments import arg_parser
from models.models import ResnetEncoderWithTarget

from wrappers.target_generator import RandomFigure, DatasetFigure
from wrappers.common_wrappers import  VectorObservationWrapper,  \
    Discretization,  flat_action_space,ColorWrapper

from wrappers.multitask import Multitask
from wrappers.reward_wrappers import RangetRewardFilledField
from wrappers.loggers import SuccessRateWrapper,Statistics,R1_score
from gridworld.env import GridWorld
from visual import Visual



def make_iglu(*args, **kwargs):
    custom_grid = np.ones((9,11,11))
    env = GridWorld(custom_grid, render=True, select_and_place=True, max_steps= 10050)
    env = Visual(env)    #
    figure_generator = RandomFigure
    env = Multitask(env,  True, True,fig_generator = figure_generator)
    env = VectorObservationWrapper(env)
    env = Discretization(env, flat_action_space('human-level'))
    env = ColorWrapper(env)
    env = RangetRewardFilledField(env)
    env = VideoLogger(env)
    env = R1_score(env)
    if isinstance(figure_generator, DatasetFigure):
        env = Statistics(env, st_name = "test.csv")
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
                           '--train_dir=../train_dir/0012'], evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())