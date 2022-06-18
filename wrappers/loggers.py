import datetime
import logging
import os
import pickle
import uuid

import cv2
import gym
import numpy as np

from wrappers.common_wrappers import Wrapper

logger = logging.getLogger(__file__)
IGLU_ENABLE_LOG = os.environ.get('IGLU_ENABLE_LOG', '')


class VideoLogger(Wrapper):
    def __init__(self, env, every=50):
        super().__init__(env)
        runtime = timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        self.dirname = f'action_logs/run-{runtime}'
        self.every = every
        self.filename = None
        self.running_reward = 0
        self.actions = []
        self.flushed = False
        self.new_session = True
        self.add_to_name = ''
        self.info = {'done': 0}
        os.makedirs(self.dirname, exist_ok=True)

    def flush(self):
        if self.filename is not None:
            self.out.release()
            os.rename(f'{self.filename}.mp4', f'{self.filename}_{self.add_to_name}_{self.env.name}.mp4')
            self.obs = []
            self.new_session = True
        if True or self.info['done'] != 'full' and self.info['done'] != 'right_move':
            timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            uid = str(uuid.uuid4().hex)
            r = np.random.randint(0, 11)
            name = f'episode-{r}_{timestamp}-{uid}'
            self.filename = os.path.join(self.dirname, name)
            self.running_reward = 0
            self.flushed = True
            self.actions = []
            self.frames = []
            self.obs = []
            self.out = cv2.VideoWriter(f'{self.filename}.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                       20, (self.env.size, self.env.size))
            self.new_session = False

    def reset(self):
        self.steps = 0
        if not self.flushed:
            self.flush()
        return super().reset()

    def close(self):
        if not self.flushed:
            self.flush()
        return super().close()

    def step(self, action):
        # assuming dict
        self.flushed = False
        obs, reward, done, info = super().step(action)
        self.info = info
        self.steps += 1
        self.actions.append(action)
        image = None
        if 'obs' in obs:
            image = np.transpose(obs['obs'], (0, 1, 2)) * 255
        elif 'obs' in info:
            image = info['obs'] * 255
        self.add_to_name = info['done']
        if image is None:
            raise Exception("No image in observation!")
        image = image[:, :, ::-1].astype(np.uint8)
        self.out.write(image)
        self.obs.append({k: v for k, v in obs.items() if k != 'obs'})
        self.obs[-1]['reward'] = reward
        self.running_reward += reward
        return obs, reward, done, info


class Logger(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        runtime = timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        self.dirname = f'action_logs/run-{runtime}'
        self.filename = None
        self.running_reward = 0
        self.actions = []
        self.flushed = False
        os.makedirs(self.dirname, exist_ok=True)

    def flush(self):
        if self.filename is not None:
            with open(f'{self.filename}-act.pkl', 'wb') as f:
                pickle.dump(self.actions, f)
            with open(f'{self.filename}-obs.pkl', 'wb') as f:
                pickle.dump(self.obs, f)
            self.obs = []
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        uid = str(uuid.uuid4().hex)
        name = f'episode-{timestamp}-{uid}'
        self.filename = os.path.join(self.dirname, name)
        self.running_reward = 0
        self.flushed = True
        self.actions = []
        self.frames = []
        self.obs = []

    def reset(self):
        if not self.flushed:
            self.flush()
        return super().reset()

    def close(self):
        if not self.flushed:
            self.flush()
        return super().close()

    def step(self, action):
        self.flushed = False
        new_action = {}
        obs, reward, done, info = super().step(action)
        self.actions.append(action)
        self.obs.append({k: v for k, v in obs.items() if k != 'pov'})
        self.obs[-1]['reward'] = reward
        self.running_reward += reward
        return obs, reward, done, info


def cubes_coordinates(grid):
    z, x, y = np.where(grid > 0)
    points = []
    for idx in range(len(z)):
        points.append([x[idx], y[idx], z[idx]])
    return points


class SuccessRateWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.sr = 0
        self.tasks_count = 0

    def reset(self):
        self.sr = 0
        self.tasks_count = 0
        return super().reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        info['episode_extra_stats'] = info.get('episode_extra_stats', {})
        roi = info['grid'][info['target_grid'] != 0]
        blocks = np.where(roi != 0)
        if reward > 1 or done:
            self.tasks_count += 1
            if reward > 1:
                self.sr += 1
            info['episode_extra_stats']['SuccessRate'] = self.sr / self.tasks_count
        return observation, reward, done, info


class SuccessRateFullFigure(gym.Wrapper):

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        info['episode_extra_stats'] = info.get('episode_extra_stats', {})
        # roi = info['grid'][info['target_grid'] != 0]
        # blocks = np.where(roi != 0)
        if done:
            if info['done'] == 'full':
                info['episode_extra_stats']['CoplitedRate'] = 1
            else:
                info['episode_extra_stats']['CoplitedRate'] = 0
        return observation, reward, done, info


class R1_score(gym.Wrapper):

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        info['episode_extra_stats'] = info.get('episode_extra_stats', {})
        # roi = info['grid'][info['target_grid'] != 0]
        # blocks = np.where(roi != 0)
        if done:
            obs_grid = info['done_grid']
            grid = np.zeros_like(obs_grid)
            target = np.zeros_like(self.env.original)
            grid[obs_grid != 0] = 1
            target[self.env.original != 0] = 1

            maximal_intersection = (grid * target).sum()
            current_grid_size = grid.sum()
            target_grid_size = target.sum()
            curr_prec = maximal_intersection / target_grid_size
            curr_rec = maximal_intersection / max(current_grid_size, 1)

            if maximal_intersection == 0:
                curr_f1 = 0
            else:
                curr_f1 = 2 * curr_prec * curr_rec / (curr_rec + curr_prec)

            if (target_grid_size == current_grid_size) and not self.env.modified and self.env.rp:
                curr_f1 = 1
                maximal_intersection = target_grid_size
                current_grid_size = target_grid_size

            info['episode_extra_stats']['R1_score'] = curr_f1
            info['episode_extra_stats']['maximal_intersection'] = maximal_intersection
            info['episode_extra_stats']['target_grid_size'] = target_grid_size
            info['episode_extra_stats']['current_grid_size'] = current_grid_size

            # maximal_intersection = (grid * target).sum()
        return observation, reward, done, info
