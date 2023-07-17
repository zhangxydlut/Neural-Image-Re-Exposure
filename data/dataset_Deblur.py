"""
Dataset for training Pixel Controlling

"""
import os
import math
import random
from glob import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from data.dataset_FECube import FECubeDataset, create_train_dataloader

import logging
logger = logging.getLogger('base')


class DeblurGoProDataset(FECubeDataset):
    def choose_source(self, sample_path, clear_only=False):
        impath_list = glob(os.path.join(sample_path, 'gs_*.png'))

        im_path = random.choice(impath_list)  # random select a rs frame # TODO: allow multiple frames
        W, H = Image.open(im_path).size
        im_name = os.path.basename(im_path)
        im_type, rs_start, rs_end, blur = im_name.replace('.png', '').split('_')
        rs_start, rs_end, blur = int(rs_start), int(rs_end), int(blur)
        assert rs_start == rs_end  # for classical deblur, the image is expected to be global shutter
        vg_name = "vg_{}x{}_{}to{}_{}bins.npz".format(H, W, rs_start, rs_end+blur-1, self.n_bins)
        vg_path = os.path.join(sample_path, vg_name)  # path to the selected voxel grid

        rs_info = dict(
            im_type=im_type,
            im_path=im_path,
            im_start=rs_start,
            im_end=rs_end,
            blur=blur,
            H=H,
            W=W,
            vg_path=vg_path,  # the time spectrum of rs is calculated relative to the vg time coordinate
            vg_start=rs_start,
            vg_end=rs_end+blur-1
        )
        vg_info = dict(
            vg_path=vg_path,  # the time spectrum of rs is calculated relative to the vg time coordinate
            vg_start=rs_start,
            vg_end=rs_end+blur-1,
            vg_bins=self.n_bins
        )
        return [rs_info], vg_info


class DeblurGoProTestset(DeblurGoProDataset):
    """
    Testing data loader for RGBE deblur on GoPro
    Following EFNet
    """
    blur_candidates = {
        11: 'gs_000000_000000_11.png',
    }

    def __init__(self, opt):
        super(DeblurGoProTestset, self).__init__(opt)
        self.data_root = opt['data_root']
        self.task = ['deblur']
        self.blur = opt['blur']  # specify which degree of blur is expected to be handled and evaluated
        self.clear_only = False  # whether to test with clear rs image
        self.n_tgt = 1

        # crop testing
        self.crop_region = None  # (320, 180, 960, 540)

        self.sample_meta = []  # meta information of each sample
        seq_dirs = [os.path.join(self.data_root, seq_name) for seq_name in os.listdir(self.data_root)]
        for seq_dir in tqdm(seq_dirs):
            self.sample_meta.extend(sorted(os.path.join(seq_dir, clip_name) for clip_name in os.listdir(seq_dir)))
        self.len = len(self.sample_meta)

    def choose_source(self, sample_path, clear_only=False):
        # select the rs to be identical to Gev-RS official
        im_path = os.path.join(sample_path, self.blur_candidates[self.blur])
        W, H = Image.open(im_path).size
        im_name = os.path.basename(im_path)
        im_type, rs_start, rs_end, blur = im_name.replace('.png', '').split('_')
        rs_start, rs_end, blur = int(rs_start), int(rs_end), int(blur)
        vg_name = "vg_{}x{}_{}to{}_{}bins.npz".format(H, W, rs_start, rs_end+blur-1, self.n_bins)
        vg_path = os.path.join(sample_path, vg_name)  # path to the selected voxel grid

        rs_info = dict(
            im_type=im_type,
            im_path=im_path,
            im_start=rs_start,
            im_end=rs_end,
            H=H,
            W=W,
            blur=blur,
            vg_path=vg_path,  # the time spectrum of rs is calculated relative to the vg time coordinate
            vg_start=rs_start,
            vg_end=rs_end+blur-1
        )
        vg_info = dict(
            vg_path=vg_path,  # the time spectrum of rs is calculated relative to the vg time coordinate
            vg_start=rs_start,
            vg_end=rs_end+blur-1,
            vg_bins=self.n_bins
        )
        return [rs_info], vg_info

    def choose_target(self, im_info_list):
        im_info = [im_info_list[(self.blur-1)//2]]
        return im_info

    def __getitem__(self, index):
        data_dict = super(DeblurGoProTestset, self).__getitem__(index)

        if self.crop_region:
            x1, y1, x2, y2, = self.crop_region
            data_dict['src_img'] = data_dict['src_img'][:, :, y1:y2, x1:x2]
            data_dict['src_tspec'] = data_dict['src_tspec'][:, :, y1:y2, x1:x2]
            data_dict['vg'] = data_dict['vg'][:, y1:y2, x1:x2]
            data_dict['vg_tspec'] = data_dict['vg_tspec'][:, y1:y2, x1:x2]
            data_dict['target_gs_imgs'] = data_dict['target_gs_imgs'][:, :, y1:y2, x1:x2]
            data_dict['tgt_tspec'] = data_dict['tgt_tspec'][:, :, y1:y2, x1:x2]

            data_dict['sample_path'] = self.sample_meta[index]
            data_dict['sample_id'] = '{}-{}'.format(*data_dict['sample_path'].split('/')[-2:])

        return data_dict


class DeblurGoProTestset360p(DeblurGoProTestset):
    blur_candidates = {
        17: 'gs_000000_000000_17.png',
        15: 'gs_000001_000001_15.png',
        13: 'gs_000002_000002_13.png',
        11: 'gs_000003_000003_11.png',
        9: 'gs_000004_000004_9.png',
        7: 'gs_000005_000005_7.png'
    }

    def choose_target(self, im_info_list):
        im_info = [im_info_list[(self.blur-1)//2]]
        return im_info

    def __getitem__(self, index):
        data_dict = super(DeblurGoProTestset, self).__getitem__(index)
        data_dict['sample_path'] = self.sample_meta[index]
        data_dict['sample_id'] = '{}-{}'.format(*data_dict['sample_path'].split('/')[-2:])
        return data_dict
