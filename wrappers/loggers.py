import datetime
import logging
import os
import pickle
import uuid

import cv2
import gym
import numpy as np
import pandas as pd

from wrappers.common_wrappers import Wrapper

logger = logging.getLogger(__file__)
IGLU_ENABLE_LOG = os.environ.get('IGLU_ENABLE_LOG', '')


class Statistics(Wrapper):
    def __init__(self, env, st_name="iglu_dataset.csv"):
        super().__init__(env)
        self.statisctics = pd.DataFrame(columns=['TaskName', 'SuccessRate',
                                                 'FDone', 'FFail', 'LastBlock', 'LastTask', 'LastAction', 'TotalTry',
                                                 'BlockInFig', 'BlocksDo', 'Complete', 'RightPredicted',
                                                 "IsModified", "R1_score", "r1sum", "maximal_intersection",
                                                 "target_grid_size", "current_grid_size", "f1_onstart"])
        self.statisctics.set_index('TaskName', inplace=True)
        self.info = dict()
        self.st_name = st_name
        self.last_action = 17

    def reset(self):
        if 'done' in self.info:
            task_name = self.env.name
            fig_done = 0
            if self.info['done'] == 'full':
                fig_done = 1
            binary = ['attack', 'forward', 'back', 'left', 'right', 'jump', 'camera', 'camera', 'camera', 'camera',
                      'MOVE']
            if self.last_action >= len(binary):
                act = 'MOVE'
            else:
                act = binary[self.last_action]

            # print(self.env.last_target)
            ltarget = np.where(self.env.task.target_grid != 0)
            ltarget = (ltarget[0], ltarget[1], ltarget[2], self.env.task.target_grid.sum())
            if self.statisctics['TotalTry'].mean() >= 10:
                self.st_name = "next_stage.csv"
            try:
                lblock = self.env.new_blocks[-1]
            except:
                lblock = -1

            try:
                sr = self.statisctics.loc[task_name]['SuccessRate']
                fdone = self.statisctics.loc[task_name]['FDone']
                ffail = self.statisctics.loc[task_name]['FFail']
                ttry = self.statisctics.loc[task_name]['TotalTry']
                fblock = self.statisctics.loc[task_name]['BlockInFig']
                blockdo = self.statisctics.loc[task_name]['BlocksDo']
                compl = self.statisctics.loc[task_name]['Complete']
                rp = self.statisctics.loc[task_name]['RightPredicted']
                rssum = self.statisctics.loc[task_name]['r1sum']
            #  modified = 0
            except:
                sr = 0
                fdone = 0
                ffail = 0
                ttry = 0
                fblock = np.sum(self.env.relief_map) - np.sum(self.env.hole_map)
                builded = len(np.where(self.env.current_grid != 0)[0])
                blockdo = builded
                compl = blockdo / fblock
                rp = self.env.rp
                rssum = 0

            f1_onstart = float(self.env.f1_onstart)

            rssum += self.info['episode_extra_stats']['R1_score']
            print(
                "....................................................................................................")
            print(rssum)
            print(self.info['episode_extra_stats']['maximal_intersection'])
            print(self.info['episode_extra_stats']['target_grid_size'])
            print(self.info['episode_extra_stats']['current_grid_size'])
            print(
                "....................................................................................................")

            modified = self.env.modified
            builded = len(np.where(self.env.current_grid != 0)[0])
            blockdo = builded
            compl += blockdo / fblock
            compl = compl / 2

            nttry = ttry + 1
            r1score = rssum / nttry

            if fig_done:
                fdone += 1
            else:
                ffail += 1

            nsr = fdone / nttry
            maximal_intersection = self.info['episode_extra_stats']['maximal_intersection']
            target_grid_size = self.info['episode_extra_stats']['target_grid_size']
            current_grid_size = self.info['episode_extra_stats']['current_grid_size']
            self.statisctics.loc[task_name] = [nsr, fdone, ffail, lblock, ltarget,
                                               act, nttry, fblock, blockdo, compl,
                                               rp, modified, r1score, rssum, maximal_intersection,
                                               target_grid_size, current_grid_size, f1_onstart]
            self.statisctics.to_csv(self.st_name)

        return super().reset()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.info = info
        self.last_action = action
        #  new_obs = obs.copy()
        # obs['chat'] = self.env.chat
        # new_obs['chat'] = self.env.chat
        # print("new_obs", new_obs.keys())
        return obs, reward, done, info


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
        # print(self.env.agent_win)
        # if not self.env.agent_win:
        # if self.info['done'] == 'full' and self.info['done'] != 'right_move':
        if self.filename is not None:
            # with open(f'{self.filename}-r{self.running_reward}.json', 'w') as f:
            #  json.dump(self.actions, f)
            self.out.release()
            os.rename(f'{self.filename}.mp4', f'{self.filename}_{self.add_to_name}_{self.env.name}.mp4')
            # with open(f'{self.filename}-obs.pkl', 'wb') as f:
            # pickle.dump(self.obs, f)
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
        new_action = {}
        obs, reward, done, info = super().step(action)
        self.info = info
        self.steps += 1
        self.actions.append(action)
        #   print((obs))
        if 'obs' in obs:
            image = np.transpose(obs['obs'], (0, 1, 2)) * 255
        elif 'obs' in info:
            image = info['obs'] * 255
        self.add_to_name = info['done']
        font = cv2.FONT_HERSHEY_SIMPLEX  # org
        org = (8, 8)  # fontScale
        fontScale = 0.3  # Blue color in BGR
        color = (0, 0, 255)  # Line thickness of 2 px
        thickness = 1

        image = image[:, :, ::-1].astype(np.uint8)
        if True:
            target = np.where(obs['target_grid'] != 0)
            if obs['target_grid'][target] > 0:
                act = 'Move block'
            else:
                act = 'Remove block'
            image = cv2.putText(image, f"{act} \\n-{target}", org, font,
                                fontScale, color, thickness, cv2.LINE_AA)

            org = (8, 18)
            binary = ['attack', 'forward', 'back', 'left', 'right', 'jump', 'camera', 'camera', 'camera', 'camera',
                      'MOVE']
            if action >= len(binary):
                act = 'MOVE'
            else:
                act = binary[action]
            image = cv2.putText(image, f"I do - {act} - {action}", org, font,
                                fontScale, color, thickness, cv2.LINE_AA)
            org = (200, 28)
            color = (0, 255, 0)
            image = cv2.putText(image, f"steps- {self.steps}", org, font,
                                fontScale, color, thickness, cv2.LINE_AA)

            org = (50, 208)
            color = (0, 255, 0)
            image = cv2.putText(image, f"inventory- {obs['inventory']}", org, font,
                                fontScale, color, thickness, cv2.LINE_AA)

            org = (50, 158)
            color = (0, 255, 0)
            image = cv2.putText(image, f"blocks - {len(np.where(obs['grid'] != 0)[0])}", org, font,
                                fontScale, color, thickness, cv2.LINE_AA)

            # org = (50, 188)
            # color = (255, 0, 255)
            # builded = len(np.where(self.env.current_grid != 0)[0])
            # need_to_do = np.sum(self.env.relief_map) - np.sum(self.env.hole_map)
            # image = cv2.putText(image, f"progress - {builded} / {need_to_do}", org, font,
            #                     fontScale, color, thickness, cv2.LINE_AA)

        # print(image.shape)
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
