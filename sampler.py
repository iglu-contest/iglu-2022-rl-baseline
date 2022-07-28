import sys
import gym
import traceback
from sample_factory.algorithms.utils.arguments import parse_args
from sample_factory.envs.env_registry import global_env_registry
from sample_factory.run_algorithm import run_algorithm
from sample_factory_examples.train_custom_env_custom_model import override_default_params_func


def make_custom_multi_env_func(full_env_name, cfg=None, env_config=None):
    import gridworld
    import gym
    env = gym.make('IGLUGridworldVector-v0')
    from gridworld.tasks import DUMMY_TASK
    env.set_task(DUMMY_TASK)
    return env

def make_custom_multi_env_func_vis(full_env_name, cfg=None, env_config=None):
    import gridworld
    import gym
    if env_config is None:
        env_config = {}
    env = gym.make('IGLUGridworld-v0', fake=env_config.get('fake', False))
    from gridworld.tasks import DUMMY_TASK
    env.set_task(DUMMY_TASK)
    return env

def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix='IGLUGridworldVector-v0',
        make_env_func=make_custom_multi_env_func,
        override_default_params_func=override_default_params_func,
    )
    global_env_registry().register_env(
        env_name_prefix='IGLUGridworld-v0',
        make_env_func=make_custom_multi_env_func_vis,
        override_default_params_func=override_default_params_func,
    )


def main():
    sys.argv += ['--algo=DUMMY_SAMPLER', '--env=IGLUGridworld-v0', '--num_workers', '8', '--sampler_worker_gpus', '0', '--num_envs_per_worker', '1']
    """Script entry point."""
    register_custom_components()
    cfg = parse_args()
    status = run_algorithm(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
