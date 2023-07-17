"""
Run test and evaluation on testing sets

Usage:
1. Test
    ```
    python test_nire.py --mode=test \
    --test_opt=options/test/GoPro-VFI7.yml \
    --vis \
    --model_path=experiments/AnyPixEF5-MT/models/90000_G_AIStation.pth
    ```
"""
import os
from datetime import datetime
import argparse
import numpy as np
from tqdm import tqdm
import yaml
import options as opt_util
from utils import OrderedYaml
Loader, Dumper = OrderedYaml()

import torch
import logging

logger = logging.getLogger('base')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ext_storage', type=str, default='/media/zxy/AnyPixEF/Dataset', help='Path to external storage')
    parser.add_argument('--test_opt', type=str, default='options/test/GevRS-Unroll.yml', help="mode option")
    parser.add_argument('--data_path', type=str, default=None, help="data path for testing")
    parser.add_argument('--model_path', type=str, default="latest_G.pth", help="model parameter path")
    parser.add_argument('--vis', action="store_true", help="whether to show VFI result during testing")
    parser.add_argument('--fast_test', action="store_true", help="whether to show VFI result during testing")
    args = parser.parse_known_args()[0]

    return args


def build_opt(args):
    exp_root = os.path.abspath(os.path.join(os.path.dirname(args.model_path), '../'))
    opt_path = os.path.join(exp_root, 'cfg.yml')
    if not os.path.exists(opt_path):
        print("{} not found! loading default option".format(opt_path))
        opt_path = "options/train/videoinr/train_videoinr.yml"  # by default load VideoINR config

    # load training time options
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)

    # test time options
    opt['is_train'] = False
    opt['dist'] = False
    opt['path']['pretrain_model_G'] = args.model_path
    opt['fast_test'] = args.fast_test

    # build testing options
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), args.test_opt)), 'r') as f:
        test_opt = yaml.load(f, Loader=Loader)

    data_root = test_opt['datasets']['test']['data_root']
    if not os.path.exists(data_root):
        data_root = os.path.join(args.ext_storage, data_root.replace('datasets/', ''))
        assert os.path.exists(data_root)
        test_opt['datasets']['test']['data_root'] = data_root

    # prepare the directory for saving results
    model_name = os.path.basename(args.model_path).split('.')[0]
    exp_root = os.path.abspath(os.path.join(os.path.dirname(args.model_path), '../'))

    args.res_dir = "{}/results/{}".format(exp_root, model_name)  # output path (output gs image)
    os.makedirs(args.res_dir, exist_ok=True)
    args.vis_res_dir = None
    if args.vis:
        args.vis_res_dir = "{}/results/{}/vis".format(exp_root, model_name)
        os.makedirs(args.vis_res_dir, exist_ok=True)

    return opt, test_opt


def test_unroll_gevrs(args, opt, test_opt):
    from models.engines import build_engine
    engine = build_engine(opt)

    def inference_wrapper(data):
        # adapt the dataloader pumping content to the engine's testing interface
        data = data
        # adapt the model output to the evaluation interface
        preds = engine.test(data)
        preds = preds[:, 0]  # only one target time is chosen for evaluation
        return preds

    from utils.eval.unroll_eval.inference import EvalUnrollGevRS
    from data.dataset_Unroll import UnrollGevRSTestset

    dataset = UnrollGevRSTestset(test_opt['datasets']['test'])
    EvalUnrollGevRS(inference_wrapper, dataset, logger).eval()


def test_deblur_gopro(args, opt, test_opt):
    from models.engines import build_engine
    engine = build_engine(opt)

    def inference_wrapper(data):
        # adapt the dataloader pumping content to the engine's testing interface
        data = data
        # adapt the model output to the evaluation interface
        preds = engine.test(data)
        preds = preds[:, 0]  # only one target time is chosen for evaluation
        return preds

    from utils.eval.deblur_eval.inference import EvalDeblurGoPro

    if test_opt['datasets']['test']['name'] == 'DeblurGoProTestset':
        from data.dataset_Deblur import DeblurGoProTestset360p
        dataset = DeblurGoProTestset360p(test_opt['datasets']['test'])
    elif test_opt['datasets']['test']['name'] == 'DeblurGoProTestset720p':
        from data.dataset_Deblur import DeblurGoProTestset
        dataset = DeblurGoProTestset(test_opt['datasets']['test'])
    else:
        raise NotImplementedError

    EvalDeblurGoPro(inference_wrapper, dataset, logger, vis_res_dir=args.vis_res_dir).eval()


def test_vfi_jdfi_gopro(args, opt, test_opt):
    from models.engines import build_engine
    engine = build_engine(opt)

    def inference_wrapper(data):
        # adapt the dataloader pumping content to the engine's testing interface
        data = data
        # adapt the model output to the evaluation interface
        preds = engine.test(data)
        return preds

    from utils.eval.vfi_eval.inference import EvalVFIGoPro

    if test_opt['datasets']['test']['name'] == 'VFIGoProTestset':
        from data.dataset_VFI import VFIGoProTestset
        dataset = VFIGoProTestset(test_opt['datasets']['test'])
    else:
        raise NotImplementedError

    EvalVFIGoPro(inference_wrapper, dataset, logger, vis_res_dir=args.vis_res_dir).eval()


if __name__ == '__main__':
    args = parse_args()
    opt, test_opt = build_opt(args)  # build the option of model, ensure the model identical to training

    # set logger
    ckpt_name = os.path.basename(args.model_path).split('.')[0]
    H, W = test_opt['datasets']['test']['resolution']
    data_name = os.path.basename(args.test_opt).split('.')[0].split('-')[0]
    task = os.path.basename(args.test_opt).split('.')[0].split('-')[1]
    test_log = os.path.join(args.res_dir, "{}-test-{}-{}-{}x{}-{}blur-{}.eval".format(
        ckpt_name, task, data_name, H, W, test_opt['datasets']['test']['blur'],
        datetime.now().strftime('%y-%m-%d-%H:%M:%S')))
    if args.vis_res_dir:
        args.vis_res_dir = os.path.join(args.vis_res_dir, task)
        os.makedirs(args.vis_res_dir, exist_ok=True)
    fh = logging.FileHandler(test_log, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info(opt_util.dict2str(opt))
    logger.info(opt_util.dict2str(test_opt))

    if test_opt['script'] == 'unroll_gevrs':
        test_unroll_gevrs(args, opt, test_opt)
    elif test_opt['script'] == 'deblur_gopro':
        test_deblur_gopro(args, opt, test_opt)
    elif test_opt['script'] in ['vfi_gopro', 'jdfi_gopro']:
        test_vfi_jdfi_gopro(args, opt, test_opt)
    else:
        raise NotImplementedError(f"Testing script '{test_opt['script']}' is not implemented")
