""" Runtime module for conveniently importing
"""
import os
import yaml
from utils import OrderedYaml
Loader, Dumper = OrderedYaml()

import torch
import logging
import options as opt_util

logger = logging.getLogger('base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_opt(model_path):
    opt_file = os.path.join(model_path, 'cfg.yml')
    # load training time options
    with open(opt_file, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)

    # test time options
    opt['is_train'] = False
    opt['dist'] = False
    opt['path']['pretrain_model_G'] = os.path.join(model_path, 'models/latest.pth')
    opt['fast_test'] = False

    return opt


def build_nire_runner(model_path):
    # get model settings
    opt = build_opt(model_path)  # build the option of model, ensure the model identical to training
    logger.info(opt_util.dict2str(opt))

    # initialize the engine
    from models.engines.NIRE_engine import NIRE_Engine
    engine = NIRE_Engine(opt)
    logger.info('Engine [{:s}] is successfully built.'.format(engine.__class__.__name__))

    # expose the testing interface
    infer_fn = engine.test
    return infer_fn
