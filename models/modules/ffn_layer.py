"""
Modified from https://github.com/linjing7/VR-Baseline/blob/main/mmedit/models/backbones/sr_backbones/FGST_util.py
"""
import math
import warnings

import torch
import torch.nn.functional as F
import torch.nn as nn


def flow_warp(x, flow, interpolation='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (h, w, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def _no_grad_trunc_normal_(tensor, mean, std, a, b):
        def norm_cdf(x):
            # Computes standard normal cumulative distribution function
            return (1. + math.erf(x / math.sqrt(2.))) / 2.

        if (mean < a - 2 * std) or (mean > b + 2 * std):
            warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                          "The distribution of values may be incorrect.",
                          stacklevel=2)
        with torch.no_grad():
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)
            tensor.uniform_(2 * l - 1, 2 * u - 1)
            tensor.erfinv_()
            tensor.mul_(std * math.sqrt(2.))
            tensor.add_(mean)
            tensor.clamp_(min=a, max=b)
            return tensor

    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:

    ::

        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.
        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self):
        """Initialize weights for ResidualBlockNoBN.

        Initialization methods like `kaiming_init` are for VGG-style
        modules. For modules with residual paths, using smaller std is
        better for stability and performance. We empirically use 0.1.
        See more details in "ESRGAN: Enhanced Super-Resolution Generative
        Adversarial Networks"
        """

        def default_init_weights(module, scale=1):
            """Initialize network weights.

            Args:
                modules (nn.Module): Modules to be initialized.
                scale (float): Scale initialized weights, especially for residual
                    blocks.
            """
            from mmcv.cnn import constant_init, kaiming_init
            from mmcv.utils.parrots_wrapper import _BatchNorm

            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m, a=0, mode='fan_in', bias=0)
                    m.weight.data *= scale
                elif isinstance(m, nn.Linear):
                    kaiming_init(m, a=0, mode='fan_in', bias=0)
                    m.weight.data *= scale
                elif isinstance(m, _BatchNorm):
                    constant_init(m.weight, val=1, bias=0)

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        return self.fn(x, *args, **kwargs)


class FeedForward(nn.Module):
    # Feedforward Network
    def __init__(self, dim, num_resblocks):
        super().__init__()
        main = []
        # residual blocks
        main.append(make_layer(ResidualBlockNoBN, num_resblocks, mid_channels=dim))
        self.net = nn.Sequential(*main)

    def forward(self, x):
        out = self.net(x)
        return out
