import numpy as np

from wrappers.common_wrappers import Wrapper


def strict_reward_range():
    reward_range = [1, 0.25, 0.05, 0.001, -0.0001, -0.001, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08,
                    -0.09]
    for_long_distance = [-0.10 - 0.01 * i for i in range(50)]
    return reward_range + for_long_distance


def remove_reward_range():
    reward_range = [1, 0.0001, 0.00, 0.00, -0.0001, -0.001, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08,
                    -0.09]
    for_long_distance = [-0.10 - 0.01 * i for i in range(50)]
    return reward_range + for_long_distance


class RangetReward(Wrapper):
    def __init__(self, env, rspec=15):
        super().__init__(env)
        self.rspec = rspec

    def calc_reward(self, dist, remove=False):
        reward_range = strict_reward_range()
        remove_reward_range_ = remove_reward_range()
        try:
            if remove:
                reward = remove_reward_range_[int(dist)]
            else:
                reward = reward_range[int(dist)]
        except Exception as e:
            raise Exception(e)
        return reward

    def blocks_count(self, info):
        return np.sum(info['grid'] != 0)

    def check_goal_closeness(self, info=None, broi=None, remove=False):
        roi = np.where(self.env.task.target_grid != 0)  # y x z
        goal = np.mean(roi[1]), np.mean(roi[2]), np.mean(roi[0])
        if broi is None:
            broi = np.where(info['grid'] != 0)  # y x z
        builds = np.mean(broi[1]), np.mean(broi[2]), np.mean(broi[0])
        dist = ((goal[0] - builds[0]) ** 2 +
                (goal[1] - builds[1]) ** 2 +
                (goal[2] - builds[2]) ** 2) ** 0.5
        return self.calc_reward(dist, remove)


def calc_new_blocks(current_grid, last_grid):
    grid = np.zeros_like(current_grid)
    relief = np.zeros_like(last_grid)
    grid[current_grid != 0] = 1
    relief[last_grid != 0] = 1

    new_blocks = np.where(grid != relief)
    if len(new_blocks[0]) > 1:
         raise Exception (f"""
               Bulded more then one block! Logical error!!
               grid z_x_y- {np.where(current_grid != 0)}
               relief z_x_y- {np.where(last_grid != 0)}
               blocks z_x_y- {np.where(grid != relief)}
               """)

    return grid, relief, np.where(grid != relief)


class RangetRewardFilledField(RangetReward):
    def __init__(self, env):
        super().__init__(env)
        self.fs = True
        self.info = dict()
        self.last_obs = None

    def reset(self):
        self.fs = True
        self.SR = 0
        self.steps = 0
        self.tasks_count = 1
        return super().reset()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        
       
       # print("from env: ", action)
        if done:
            info['done'] = 'len_done_%s' % self.steps
        info['done'] = 'len_done_%s' % self.steps
        self.last_obs = obs
        self.steps += 1

        ### Calc count of new blocks
        grid, relief, new_blocks = calc_new_blocks(obs['grid'], self.env.current_grid)
        info['done_grid'] = grid
        info['episode_extra_stats'] = info.get('episode_extra_stats', {})

        ### Reward calculation
        reward = 0
        task = np.sum(self.env.task.target_grid)  # if < 0 - task if remove, else task is build
        
        ### Move or rebove block? 
        if 5 < action < 12 or action == 17:
            do = 1
            full = self.env.one_round_reset(new_blocks, do)
        elif action == 16:
            do = 0
            full = self.env.one_round_reset(new_blocks, do)
            
        if len(new_blocks[0]) >= 1:
            grid_block_count = len(np.where(grid != 0)[0])
            relief_block_count = len(np.where(relief != 0)[0])

            if task < 0 and grid_block_count > relief_block_count:  # если нужно удалить кубик, а агент его поставил
                reward = -0.001
            elif task > 0 and grid_block_count < relief_block_count:  # если нужно поставить кубик, а агент его удалил
                reward = -0.001
            else:
                reward = self.check_goal_closeness(info, broi=new_blocks, remove=task < 0)  # иначе

            
           # print(action)

            ### Add reward for block under agent
            x_agent, z_agent, y_agent = obs['agentPos'][:3]
            x_agent, y_agent = x_agent + 5, y_agent + 5
            x_agent, y_agent = int(x_agent + 0.5), int(y_agent + 0.5)
            z_last_block, x_last_block, y_last_blcok = np.where(self.env.task.target_grid != 0)
            if reward == 1:
                self.SR += 1
                if x_last_block == x_agent and y_last_blcok == y_agent and (z_agent - z_last_block) <= 2:
                    reward += 0.5
                if task < 0:
                    if int(x_last_block - x_agent) >= 0 and int(
                            y_last_blcok - y_agent) >= 0 and z_agent >= z_last_block:
                        #     raise Exception("WRONG!")
                        reward += 0.5
                #full = self.env.one_round_reset(new_blocks, do)
                #print("Success")
                info['done'] = 'right_move'
                if full:
                    info['done'] = 'full'
                    done = True

            if reward < 1:
                info['done'] = 'mistake_%s' % self.steps
               # done = True
               # full = self.env.one_round_reset(new_blocks, do)
                self.env.update_field(new_blocks, do)
                if full:
                    info['done'] = 'full'
                    done = True
            if done:
                info['episode_extra_stats']['SuccessRate'] = self.SR / self.tasks_count
            self.tasks_count += 1
        self.last_grid = obs['grid']
        self.fs = False
        self.info = info
        return obs, reward, done, info


class Closeness(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.dist = 1000000

    def reset(self):
        self.dist = 1000000
        return super().reset()

    def closeness(self, info):

        roi = np.where(self.env.task.target_grid != 0)  # y x z
        goal = np.mean(roi[1]), np.mean(roi[2]), np.mean(roi[0])
        agent = info['agentPos'][:3]
        agent_pos = agent[0] + 5, agent[2] + 5, agent[1] + 1

        dist = ((goal[0] - agent_pos[0]) ** 2 + (goal[1] - agent_pos[1]) ** 2 + (goal[2] - agent_pos[2]) ** 2) ** 0.5
        return dist

    def calc_reward(self, info):
        d2 = self.closeness(info)
        if d2 < self.dist:
            self.dist = d2
            return 0.001
        else:
            return 0

    def step(self, action):
        obs, reward, done, info = super().step(action)
        add_reward = self.calc_reward(info)
        reward += add_reward
        return obs, reward, done, info
