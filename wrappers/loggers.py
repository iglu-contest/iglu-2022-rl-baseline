import gym
import os
import datetime
import pickle
import uuid
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from wrappers.common_wrappers import Wrapper
from wrappers.artist import drow_circle
logger = logging.getLogger(__file__)
IGLU_ENABLE_LOG = os.environ.get('IGLU_ENABLE_LOG', '')



class ActLoggerFull(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.paths = []
        runtime = timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        self.dirname = f'pics_logs/run-{runtime}'
        self.filename = None
        self.pic = np.zeros((11, 11, 3))
        self.prebuils = np.zeros((11,11))
        self.path = []
        self.flushed = False
        self.info = dict()
        os.makedirs(self.dirname, exist_ok=True)

    def draw_picture(self):
        print("DRAW NEW PIC!")
        ### BASE
        self.pic = np.zeros((11, 11, 3))
        self.pic[:, :, 0] = 58 / 255
        self.pic[:, :, 1] = 56 / 255
        self.pic[:, :, 2] = 39 / 255

        ### Total target
       # X,Y = np.where(self.env.relief_map!=0)
        relief = self.env.relief_map
        X,Y = np.where(relief!=0)
        self.pic[X, Y, 0] += 0
        self.pic[X, Y, 1] += 0
        self.pic[X,Y,2]  += 0.15

        ### Prebuilds
    #    print(self.env.preinited_grid.shape)
   #     print(self.pic.shape)
        self.prebuils = self.env.preinited_grid.sum(axis = 0)
    #    prebuilds_map = self.prebuils.sum(axis = 0)
    # print(prebuilds_map.shape)
        self.pic[self.prebuils!=0, 0] = self.prebuils[self.prebuils!=0] / 9
        self.pic[self.prebuils != 0, 1] = self.prebuils[self.prebuils != 0] / 9
        self.pic[self.prebuils != 0, 2] = self.prebuils[self.prebuils != 0] / 9

        ### Builds
        pgrid = self.env.preinited_grid[:,:,:].sum(axis = 0)
        lgrid = self.env.old_grid[:,:,:].sum(axis = 0)
    #    print()
    #    print(" --------------  ----------------- -----------------")
    #    print("preinited: ",pgrid)
    #    print("cgrid: ", pgrid)
    #    print("lgrid: ", lgrid)
    #    print(" --------------  ----------------- -----------------")
    #    print()
        new_blocks = np.zeros_like(pgrid)
        new_blocks = lgrid - pgrid
       # new_blocks[cgrid[pgrid!=0]!=0] = cgrid[pgrid!=0]


        blocks = self.env.new_blocks
        if len(blocks) > 0:
            for block in blocks:
                z, x,y, b = block
                if b == 1:
                    self.pic[x, y, :] = lgrid[x,y]/6


        zt, xt, yt = self.env.last_target
        self.pic[xt, yt, 0] = 1
        self.pic[xt, yt, 1] = 0
        self.pic[xt, yt, 2] = 0

        if len(blocks) > 0:
            z, x, y, b = blocks[-1]
         #   print("Final block - ",x, y)
          #  x,y = Xb[-1], Yb[-1]
            if xt == x and yt == y:
                if b == 1:
                    self.pic[x, y] = 1
                if b == 0:
                    self.pic[x, y, 2] = 1
                    self.pic[x, y, 0] = 0
                    self.pic[x, y, 1] = 0
            else:
                self.pic[x, y] = 0


            # if len(new_blocks[:-1])>0:
        #    self.pic[x, y] = 0

        ### Full path calculation
        new_path = []
        one_block = 256 / 11
      #  for j, path in enumerate(self.paths[-1]):
        for j in range(1):
            one_block = 256 / 11

            for i, coor in enumerate(self.paths[-1]):
                xp, yp = coor
                newxp = int(xp * one_block + (one_block // 2))
                newyp = int(yp * one_block + (one_block // 2))
               # print("path - ", newyp, newxp)
                new_path.append((newxp, newyp, 4, 1, 1))
            if len(new_path)>0:
                new_path.append((newxp, newyp, 8, 1, 1))

       # zt, xt, yt = self.env.last_target
     #   xt += 5
     #   yt += 5
        newxp = int(xt * one_block + (one_block // 2))
        newyp = int(yt * one_block + (one_block // 2))
        ### Draw path
        pic = cv2.resize(self.pic.copy(), (256, 256), interpolation=cv2.INTER_NEAREST)
        for i, paramentrs in enumerate(new_path):
                x,y,R,color, alpha = paramentrs
                mask = drow_circle(pic[:, :, 0], R, (x,y))
                pic[mask, 0] = 0
                pic[mask, 1] = 0
                pic[mask, 2] = 0
                pic[mask, color] = alpha
        mask = drow_circle_line(pic[:, :, 2],  10, (newxp, newyp))

        pic[mask, 0] = 0
        pic[mask, 1] = 0
        pic[mask, 2] = 1

        return pic

    def flush(self):

       # timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        self.num +=1
      #  uid = str(uuid.uuid4().hex)
        name = f'episode-{self.num}'

        self.filename = os.path.join(self.one_episode_slides_dir, name)
        pic = self.draw_picture()

        plt.imsave(self.filename + self.info['done']+ ".png", pic)

        self.flushed = True
        self.first_cube = None
        self.blocks = None


    def reset(self):
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        a = np.random.randint(0,10)
        os.makedirs(self.dirname+"/%s-%s"%(timestamp, a), exist_ok=True)
        self.one_episode_slides_dir = os.path.join(self.dirname, "%s-%s"%(timestamp,a))

        self.num = 0
        self.paths = []
        self.path = []
        self.info = dict()
        return self.env.reset()

    def close(self):
       # if not self.flushed:
            #self.flush()
        return self.env.close()

    def step(self, action):
        self.flushed = False
        obs, reward, done, info = super().step(action)
        self.info = info
        x,y = obs['agentPos'][0], obs['agentPos'][1]
       # if len(self.path) == 0:
       #     self.prebuils = np.sum(self.current_grid, axis = 0)
        pose = obs['agentPos'][:3:2] + 5
     #   print("POSE of agent", pose, obs['agentPos'])
        if (pose[0] > 0 and pose[0] <= 10) and (pose[1] > 0 and pose[1] <= 10):
            self.path.append((int(pose[0] + 0.5), int(pose[1] + 0.5)))

        if done or reward>=1:
            self.paths.append(self.path)
          #  print("PATHS: ", self.paths)
            self.path = []
       # print(self.info)
        if info['done']=='right_move' or info['done']=='full':
            self.flush()
        if done:
                self.flush()
        self.info = info
        return obs, reward, done, info


class ActLogger(Wrapper):
    def __init__(self, env, every=50):
        super().__init__(env)
        runtime = timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        self.dirname = f'pics_logs/run-{runtime}'
        self.every = every
        self.filename = None
        self.pic = np.zeros((11, 11, 3))
        self.running_reward = 0
        self.actions = []
        self.info = dict()
        self.flushed = False
        self.goal = [0,0]
        self.episode_num = 0
        self.fs = True
        os.makedirs(self.dirname, exist_ok=True)

    def flush(self):
        if self.filename is not None:
            grid = np.where(self.info['grid'] != 0)
            for zg, xg, yg in zip(*grid):
                color = 0.2 * (zg+1)
                self.pic[xg,yg] = color
                pic = cv2.resize(self.pic.copy(), (256, 256), interpolation=cv2.INTER_NEAREST)
                # print(self.first_cube)
                plt.imsave(self.filename + "-grid.png", pic)
                # застройка
            print(" \n -----blocks", self.blocks)
            X,Y = self.blocks
            x, y = self.goal

            one_block = 256/11
            path_pic = np.zeros_like(self.pic)
            new_path = []
            for i, coor in enumerate(self.path):
                xp, yp = coor
                newxp = int(xp * one_block + (one_block // 2) )
                newyp = int(yp * one_block + (one_block // 2) )
                new_path.append((newxp,newyp))
                #if i==0:
                #    path_pic[xp, yp, 1] = 0.9
                #else:
                 #   path_pic[xp, yp, 1] = 0.6


            if len(X)>=1 and (X[0]==x and Y[0]==y):
                self.pic[X[0], Y[0], :] = 1
                print(" \n -----wb")
            else:
               # self.pic[X,Y,2] = 1
                if len(X)>0:
                    self.pic[X,Y,:] = 0
                self.pic[x, y, 0] = 0.8  # цель


            #relief = self.fake_grid.mean(axis = 0)
          #  self.pic[relief!=0]=0.2
           # if self.first_cube is not None:
             #   xf, yf = self.first_cube
             #   self.pic[self.first_cube] = 0  # первый кубик
               # if x == xf and y == yf:
                  #  print(" \n -----wb")
                  #  self.pic[self.first_cube] = 1

            pic = cv2.resize(self.pic.copy(), (256, 256),  interpolation =  cv2.INTER_NEAREST)
            for i, coor in enumerate(new_path):
             #   print(coor)

                if i == 0:
                    mask = drow_circle(pic[:, :, 0], 2, coor)
                    pic[mask,1] = 0.9
                elif i == len(new_path)-1:
                    mask = drow_circle(pic[:, :, 0], 9, coor)
                    pic[mask, 2] = 0.6
                else:
                    mask = drow_circle(pic[:, :, 0], 4, coor)
                    pic[mask,1] = 0.6

            # print(self.first_cube)
            plt.imsave(self.filename + ".png", pic)
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        self.flushed = True
        self.first_cube = None
        self.blocks = None
        self.info = dict()
        self.episode_num+=1
        uid = str(uuid.uuid4().hex)
        name = f'episode{self.episode_num}-{timestamp}-{uid}'
        self.filename = os.path.join(self.dirname, name)
        self.pic = np.zeros((11, 11, 3))
        self.pic[:, :, 0] = 69 / 255
        self.pic[:, :, 1] = 67 / 255
        self.pic[:, :, 2] = 59 / 255

    def reset(self):

        if not self.flushed:
            self.flush()
        self.info = dict()
        self.path = []
        return super().reset()

    def step(self, action):
        self.flushed = False
        obs, reward, done, info = super().step(action)
        if self.relief is not None:
           grid = np.zeros_like(obs['grid'])
           relief = np.zeros_like(obs['grid'])

           grid[obs['grid']!=0] = 1

           print("-> old relief", np.where(self.env.old_relief!=0))
           print("-> relief",np.where(self.env.relief!=0))
           if reward >= 1.39:
               relief [self.env.old_relief!=0] = 1
           else:
                relief [self.env.relief!=0] = 1
           new_blocks = np.where(grid != relief)
           if len(new_blocks[0])>0:
            print("new blocks -> ", new_blocks)
           self.blocks = new_blocks[1:]

         #  print("blocks in logger", self.blocks)
    #    print(np.where(obs['target_grid']!=0))
        if len(self.info.keys())==0:
            self.info['target_grid'] = obs['target_grid']
            x, y = np.where(self.info['target_grid'] != 0)[1:]
            self.goal = [x,y]
        self.info = info
        pose = info['agentPos'][:3:2] + 5
        if (pose[0] > 0 and pose[0] <= 10) and (pose[1] > 0 and pose[1] <= 10):
            self.path.append((int(pose[0]+0.5), int(pose[1]+0.5)))
          #  self.pic[int(pose[0]), int(pose[1]), 1] = 0.4  # маршрут
        if len(np.where(info['grid'] != 0)[0]) == 1:
            self.first_cube = np.where(info['grid'] != 0)[1:]
        return obs, reward, done, info

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
        self.last_action =  17

    def reset(self):
        if 'done' in self.info:
            task_name = self.env.name
            fig_done = 0
            if self.info['done'] == 'full':
                fig_done = 1
            binary = ['attack', 'forward', 'back', 'left', 'right', 'jump', 'camera', 'camera', 'camera', 'camera', 'MOVE']
            if self.last_action >= len(binary):
                act = 'MOVE'
            else:
                act = binary[self.last_action]

            #print(self.env.last_target)
            ltarget = np.where(self.env.task.target_grid!=0)
            ltarget = (ltarget[0],ltarget[1],ltarget[2], self.env.task.target_grid.sum())
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
                compl = blockdo/fblock
                rp = self.env.rp
                rssum = 0

            f1_onstart = float(self.env.f1_onstart)

            rssum += self.info['episode_extra_stats']['R1_score']
            print("....................................................................................................")
            print(rssum)
            print(self.info['episode_extra_stats']['maximal_intersection'])
            print(self.info['episode_extra_stats']['target_grid_size'])
            print(self.info['episode_extra_stats']['current_grid_size'])
            print("....................................................................................................")

            modified = self.env.modified
            builded = len(np.where(self.env.current_grid != 0)[0])
            blockdo = builded
            compl += blockdo/fblock
            compl = compl/2


            nttry = ttry+1
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
                                               rp,modified,r1score, rssum, maximal_intersection,
                                               target_grid_size, current_grid_size,f1_onstart]
            self.statisctics.to_csv(self.st_name)

        return super().reset()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.info = info
        self.last_action = action
      #  new_obs = obs.copy()
        #obs['chat'] = self.env.chat
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
        self.info = {'done':0}
        os.makedirs(self.dirname, exist_ok=True)

    def flush(self):
       # print(self.env.agent_win)
        #if not self.env.agent_win:
        #if self.info['done'] == 'full' and self.info['done'] != 'right_move':
        if self.filename is not None:
                    # with open(f'{self.filename}-r{self.running_reward}.json', 'w') as f:
                    #  json.dump(self.actions, f)
                    self.out.release()
                    os.rename(f'{self.filename}.mp4', f'{self.filename}_{self.add_to_name}_{self.env.name}.mp4')
                    #with open(f'{self.filename}-obs.pkl', 'wb') as f:
                       # pickle.dump(self.obs, f)
                    self.obs = []
                    self.new_session = True
        if True or self.info['done']!='full' and self.info['done']!='right_move':
            timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            uid = str(uuid.uuid4().hex)
            r = np.random.randint(0,11)
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
        self.steps +=1
        self.actions.append(action)
     #   print((obs))
        if 'obs' in obs:
            image = np.transpose(obs['obs'], (1, 2, 0))
        elif 'obs' in info:
            image = info['obs'] * 255
        self.add_to_name = info['done']
        font = cv2.FONT_HERSHEY_SIMPLEX        # org
        org = (8, 8)        # fontScale
        fontScale = 0.3        # Blue color in BGR
        color = (0, 0, 255)        # Line thickness of 2 px
        thickness = 1

        image = image[:,:,::-1].astype(np.uint8)

        if True:
            target = np.where(obs['target_grid']!=0)
            if obs['target_grid'][target] > 0:
                act = 'Move block'
            else:
                act = 'Remove block'
            image = cv2.putText(image, f"{act} \\n-{target}", org, font,
                                fontScale, color, thickness, cv2.LINE_AA)


            org = (8, 18)
            binary = ['attack', 'forward', 'back', 'left', 'right', 'jump', 'camera', 'camera', 'camera', 'camera', 'MOVE']
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
            image = cv2.putText(image, f"blocks - {len(np.where(obs['grid']!=0)[0])}", org, font,
                                fontScale, color, thickness, cv2.LINE_AA)

            org = (50, 188)
            color = (255, 0, 255)
            builded = len(np.where(self.env.current_grid!=0)[0])
            need_to_do = np.sum(self.env.relief_map) - np.sum(self.env.hole_map)
            image = cv2.putText(image, f"progress - {builded} / {need_to_do}", org, font,
                                fontScale, color, thickness, cv2.LINE_AA)

       #print(image.shape)
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
                self.sr+=1
            info['episode_extra_stats']['SuccessRate'] = self.sr/self.tasks_count
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
            grid[obs_grid!=0]=1
            target[self.env.original!=0] = 1

            maximal_intersection = (grid*target).sum()
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


            #maximal_intersection = (grid * target).sum()
        return observation, reward, done, info
