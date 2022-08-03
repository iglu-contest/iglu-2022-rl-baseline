import torch
from sample_factory.algorithms.appo.model_utils import EncoderBase, \
    ResBlock, nonlinearity, get_obs_shape
from sample_factory.algorithms.utils.pytorch_utils import calc_num_elements
from torch import nn


def create_resnet_encoder(input_ch, resnet_conf, timing, cfg, use_initial_max_pooling=True):
    curr_input_channels = input_ch
    layers = []
    for i, (out_channels, res_blocks) in enumerate(resnet_conf):
        layers.append(nn.Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=1, padding=1))
        if use_initial_max_pooling:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        for j in range(res_blocks):
            layers.append(ResBlock(cfg, out_channels, out_channels, timing))

        curr_input_channels = out_channels

    layers.append(nonlinearity(cfg))

    conv_head = nn.Sequential(*layers)
    return conv_head


class ResnetEncoderWithTarget(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)
        target_shape = get_obs_shape(obs_space['target_grid'])
        rgb_shape = get_obs_shape(obs_space['obs'])
        self.rgb_encoder = create_resnet_encoder(input_ch=rgb_shape.obs[0], resnet_conf=[[16, 2], [32, 2], [32, 2]],
                                                 cfg=cfg, timing=self.timing, )
        self.target_encoder = create_resnet_encoder(input_ch=target_shape.obs[0], resnet_conf=[[64, 3]], cfg=cfg,
                                                    timing=self.timing, use_initial_max_pooling=False)
        self.inventory_compass_emb = nn.Sequential(
            nn.Linear(7, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ReLU(),
        )

        self.conv_target_out_size = calc_num_elements(self.target_encoder, target_shape.obs)
        self.conv_grid_out_size = calc_num_elements(self.rgb_encoder, rgb_shape.obs)
        self.init_fc_blocks(self.conv_target_out_size + self.conv_grid_out_size + cfg.hidden_size)

    def forward(self, obs_dict):
        max_compass_val = 360
        abs_compass_val = 180
        max_inventory_val = 20
        max_obs_value = 255

        inventory_compass = torch.cat(
            [obs_dict['inventory'] / max_inventory_val, (obs_dict['compass'] + abs_compass_val) / max_compass_val], -1)
        inv_comp_emb = self.inventory_compass_emb(inventory_compass)

        target = torch.zeros_like((obs_dict['target_grid']))
        target[obs_dict['target_grid'] > 0] = 1  # put 1 if task is build block
        target[obs_dict['target_grid'] < 0] = -1  # put -1 if task is remove block
        tg = self.target_encoder(target)
        tg_embed = tg.contiguous().view(-1, self.conv_target_out_size)

        grid = obs_dict['obs'] / max_obs_value

        grid = self.rgb_encoder(grid)
        grid_embed = grid.contiguous().view(-1, self.conv_grid_out_size)

        head_input = torch.cat([inv_comp_emb, tg_embed, grid_embed], -1)

        x = self.forward_fc_blocks(head_input)
        return x

