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
    if 'demo' in args.mode:
        seqname = os.path.basename(args.data_path)
        args.out_path_recon = "{}/demo/{}/{}/Recon/".format(exp_root, seqname, model_name)  # output path (input rs image)
        args.out_path_gs = "{}/demo/{}/{}/GT/".format(exp_root, seqname, model_name)  # output path (output gs image)
        args.out_path_rs = "{}/demo/{}/{}/INP/".format(exp_root, seqname, model_name)  # output path (output gs image)
        os.makedirs(args.out_path_recon, exist_ok=True)
        os.makedirs(args.out_path_gs, exist_ok=True)
        os.makedirs(args.out_path_rs, exist_ok=True)

    if 'test' in args.mode:
        args.res_dir = "{}/results/{}".format(exp_root, model_name)  # output path (output gs image)
        os.makedirs(args.res_dir, exist_ok=True)
        args.vis_res_dir = None
        if args.vis:
            args.vis_res_dir = "{}/results/{}/vis".format(exp_root, model_name)
            os.makedirs(args.vis_res_dir, exist_ok=True)

    return opt, test_opt


def demo_unroll_gevrs(args, opt, test_opt):
    from skimage import io
    # prepare data
    from data.dataset_Unroll import UnrollGevRSDemoset

    dataset = UnrollGevRSDemoset(test_opt['datasets']['test'], args.data_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)

    # build model
    from models.engines import build_engine
    engine = build_engine(opt)

    for i, data in tqdm(enumerate(dataloader)):
        im_recon_seq = engine.demo(data)
        rs_path = os.path.join(args.out_path_rs, "rs_{:06}.png".format(i))
        im_rs = (data['src_img'][0].clamp(0., 1.).detach().cpu()[0].permute(1, 2, 0) * 255).numpy().astype(np.uint8)
        io.imsave(rs_path, im_rs)
        for j in range(im_recon_seq.shape[1]):
            im_recon = im_recon_seq[:, j]
            im_gs = data['target_gs_imgs'][:, j]  # GT
            gs_path = os.path.join(args.out_path_gs, "gs_{:06}_{:03}.png".format(i, j))
            recon_path = os.path.join(args.out_path_recon, "recon_{:06}_{:03}.png".format(i, j))
            im_gs = (im_gs.clamp(0., 1.).detach().cpu()[0].permute(1, 2, 0) * 255).numpy().astype(np.uint8)
            im_recon = (im_recon.clamp(0., 1.).detach()[0].cpu().permute(1, 2, 0) * 255).numpy().astype(np.uint8)

            io.imsave(gs_path, im_gs)
            io.imsave(recon_path, im_recon)


def demo_deblur_gopro(args, opt, test_opt):
    from skimage import io
    # prepare data
    from data.dataset_Deblur import DeblurGoProDemoset, DeblurGoProDemoset720p

    if test_opt['datasets']['test']['name'] == 'DeblurGoProTestset':
        dataset = DeblurGoProDemoset(test_opt['datasets']['test'], args.data_path)
    elif test_opt['datasets']['test']['name'] == 'DeblurGoProTestset720p':
        dataset = DeblurGoProDemoset720p(test_opt['datasets']['test'], args.data_path)
    else:
        raise NotImplementedError
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)

    # build model
    from models.engines import build_engine
    engine = build_engine(opt)

    for i, data in tqdm(enumerate(dataloader)):
        im_recon_seq = engine.demo(data)
        im_path = os.path.join(args.out_path_rs, "blur_{:06}.png".format(i))
        im_rs = (data['src_img'][0].clamp(0., 1.).detach().cpu()[0].permute(1, 2, 0) * 255).numpy().astype(np.uint8)
        io.imsave(im_path, im_rs)
        for j in range(im_recon_seq.shape[1]):
            im_recon = im_recon_seq[:, j]
            im_gs = data['target_gs_imgs'][:, j]  # GT
            gs_path = os.path.join(args.out_path_gs, "gs_{:06}_{:03}.png".format(i, j))
            recon_path = os.path.join(args.out_path_recon, "recon_{:06}_{:03}.png".format(i, j))
            im_gs = (im_gs.clamp(0., 1.).detach().cpu()[0].permute(1, 2, 0) * 255).numpy().astype(np.uint8)
            im_recon = (im_recon.clamp(0., 1.).detach()[0].cpu().permute(1, 2, 0) * 255).numpy().astype(np.uint8)

            io.imsave(gs_path, im_gs)
            io.imsave(recon_path, im_recon)


def demo_vfi_gopro(args, opt, test_opt):
    pass


if __name__ == '__main__':
    args = parse_args()
    opt, test_opt = build_opt(args)  # build the option of model, ensure the model identical to training

    if args.mode == 'demo':
        if test_opt['script'] == 'unroll_gevrs':
            demo_unroll_gevrs(args, opt, test_opt)
        elif test_opt['script'] == 'deblur_gopro':
            demo_deblur_gopro(args, opt, test_opt)
        elif test_opt['script'] == 'vfi_gopro':
            demo_vfi_gopro(args, opt, test_opt)
