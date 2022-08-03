from collections import defaultdict

import numpy as np
import torch
from sample_factory.algorithms.appo.model_utils import EncoderBase, \
    ResBlock, nonlinearity, get_obs_shape
from sample_factory.algorithms.utils.pytorch_utils import calc_num_elements
from sample_factory.utils.timing import Timing
from torch import nn


def grid_stucture_encoder(input_channel, block_config, net_config, timing):
    layers = []
    for i, (out_channels, res_blocks) in enumerate(block_config):
        layers.extend([
            nn.Conv2d(input_channel, out_channels, kernel_size=3, stride=1, padding=1),  # padding SAME
        ])
        for j in range(res_blocks):
            layers.append(ResBlock(net_config, out_channels, out_channels, timing))
        input_ch_grid = out_channels
    layers.append(nonlinearity(net_config))
    grid_encoder = nn.Sequential(*layers)
    return grid_encoder

def obs_stucture_encoder(input_channel, block_config, net_config, timing):
    layers = []
    for i, (out_channels, res_blocks) in enumerate(block_config):
        layers.extend([
            nn.Conv2d(input_channel, out_channels, kernel_size=3, stride=1, padding=1),  # padding SAME
             nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #MAXPOOL
        ])
        for j in range(res_blocks):
            layers.append(ResBlock(net_config, out_channels, out_channels, timing))
        input_ch_grid = out_channels
    layers.append(nonlinearity(net_config))
    grid_encoder = nn.Sequential(*layers)
    return grid_encoder


class ResnetEncoderWithTarget(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)
        target_shape = get_obs_shape(obs_space['target_grid'])
        input_ch_targ = target_shape.obs[0]
        target_conf = [[64, 3]]

        grid_shape = get_obs_shape(obs_space['obs'])
        input_ch_grid = grid_shape.obs[0]
        grid_conf = [[64, 2], [64, 2], [64, 2]]
        
        ### Obs embedding
        self.conv_grid = obs_stucture_encoder(input_ch_grid, grid_conf, cfg, self.timing)
        ### Target embedding
        self.conv_target = grid_stucture_encoder(input_ch_targ, target_conf, cfg, self.timing)

        self.inventory_compass_emb = nn.Sequential(
            nn.Linear(7, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ReLU(),
        )

        self.conv_target_out_size = calc_num_elements(self.conv_target, target_shape.obs)
        self.conv_grid_out_size = calc_num_elements(self.conv_grid, grid_shape.obs)
        self.init_fc_blocks(self.conv_target_out_size + self.conv_grid_out_size + cfg.hidden_size)

    def forward(self, obs_dict):
        # values for normalization
       # abs_max_obs = np.array([10, 8, 10, 180, 360])  # x, y, z, yaw, pitch
      #  true_max_obs = np.array([5, 0, 5, 90, 0])  # x, y, z, yaw, pitch
        
        max_compass_val = 360
        abs_compass_val = 180
        max_inventory_val = 20
        max_obs_value = 255

       # abs_max_obs = torch.from_numpy(abs_max_obs).cuda()
      #  true_max_obs = torch.from_numpy(true_max_obs).cuda()

        inventory_compass = torch.cat(
            [obs_dict['inventory'] / max_inventory_val, (obs_dict['compass']+abs_compass_val)/max_compass_val], -1)
        inv_comp_emb = self.inventory_compass_emb(inventory_compass)

        target = torch.zeros_like((obs_dict['target_grid']))
        target[obs_dict['target_grid'] > 0] = 1  # put 1 if task is build block
        target[obs_dict['target_grid'] < 0] = -1  # put -1 if task is remove block
        tg = self.conv_target(target)
        tg_embed = tg.contiguous().view(-1, self.conv_target_out_size)

        grid = obs_dict['obs']/max_obs_value
        
        grid = self.conv_grid(grid)
        grid_embed = grid.contiguous().view(-1, self.conv_grid_out_size)

        head_input = torch.cat([inv_comp_emb, tg_embed, grid_embed], -1)

        x = self.forward_fc_blocks(head_input)
        return x


def main():
    def validate_config(config):
        exp = Experiment(**config)
        flat_config = Namespace(**exp.async_ppo.dict(),
                                **exp.experiment_settings.dict(),
                                **exp.global_settings.dict(),
                                **exp.evaluation.dict(),
                                full_config=exp.dict()
                                )
        return exp, flat_config

    from argparse import Namespace
    from utils.config_validation import Experiment
    from utils.create_env import make_iglu

    exp = Experiment()
    exp, flat_config = validate_config(exp.dict())
    flat_config.hidden_size = 512
    env = make_iglu()
    encoder = ResnetEncoderWithTarget(flat_config, env.observation_space, Timing())

    obs = env.reset()
    counter = defaultdict(set)
    for _ in range(10000):
        for idx, x in enumerate(list(np.round(obs[0]['agentPos'], 0))):
            counter[idx].add(int(x))

        obs, reward, done, info = env.step([env.action_space.sample()])
    obs = obs[0]

    obs['agentPos'] = torch.Tensor(obs['agentPos'])[None]
    obs['inventory'] = torch.Tensor(obs['inventory'])[None]
    obs['target_grid'] = torch.Tensor(obs['target_grid'])[None]
    obs['obs'] = torch.Tensor(obs['obs'])[None]


if __name__ == '__main__':
    main()
