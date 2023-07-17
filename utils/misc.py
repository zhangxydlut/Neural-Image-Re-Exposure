# this code is modified from https://github.com/xinntao/EDVR/blob/master/codes/utils/util.py
import os
import sys
import time
import random
import math
import socket
from datetime import datetime
import logging
from collections import OrderedDict

import numpy as np
import cv2
import torch
from torchvision.utils import make_grid
from shutil import get_terminal_size
import glob
import re

import yaml
import torch.distributed as dist
import functools

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

####################
# miscellaneous
####################
def get_model_total_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return (1.0 * params / (1000 * 1000))


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir_safe(path, allow_create=True):
    if os.path.exists(path):
        new_path = path + '_' + get_timestamp()
        print(f"{path} exists, create {new_path} instead")
        path = new_path
    if allow_create:
        os.makedirs(path)
    return path


def mkdir(path, allow_create=True):
    if not os.path.exists(path) and allow_create:
        os.makedirs(path)
    return path


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}_{}.log'.format(get_timestamp(), socket.gethostname()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()


def model_ema(self, decay=0.999):  # TODO: Exponential Model Average - have some try
    # Modified from https://github.com/XPixelGroup/BasicSR/blob/ce5c55aec0d14a399998a8c63fc650fdce3eb519/basicsr/models/base_model.py#L75
    net_g = self.get_bare_model(self.net_g)

    net_g_params = dict(net_g.named_parameters())
    net_g_ema_params = dict(self.net_g_ema.named_parameters())

    for k in net_g_ema_params.keys():
        net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)


def get_dist_info():
    # Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):  # TODO: Use decorator for convenience
    # Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def find_unused_params(model):
    # find unused parameters
    for name, param in model.named_parameters():
        if param.grad is None:
            print(name)
