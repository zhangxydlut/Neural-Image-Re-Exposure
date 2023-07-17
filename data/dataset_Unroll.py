"""
Dataset for training Pixel Controlling

"""
import os
import random
from glob import glob
from tqdm import tqdm
from PIL import Image

from data.dataset_FECube import FECubeDataset, create_train_dataloader

import logging
logger = logging.getLogger('base')


class UnrollGevRSDataset(FECubeDataset):
    def choose_source(self, sample_path, clear_only=False):
        if clear_only:
            rs_impath_list = glob(os.path.join(sample_path, 'rs_*_1.png'))
        else:
            rs_impath_list = glob(os.path.join(sample_path, 'rs_*.png'))

        rs_impath = random.choice(rs_impath_list)  # random select a rs frame # TODO: allow multiple RS frames
        W, H = Image.open(rs_impath).size
        rs_name = os.path.basename(rs_impath)
        im_type, rs_start, rs_end, blur = rs_name.strip('.png').split('_')
        rs_start, rs_end, blur = int(rs_start), int(rs_end), int(blur)
        vg_name = "vg_{}x{}_{}to{}_{}bins.npz".format(H, W, rs_start, rs_end + blur - 1, self.n_bins)
        vg_path = os.path.join(sample_path, vg_name)  # path to the selected voxel grid

        rs_info = dict(
            im_type=im_type,
            im_path=rs_impath,
            im_start=rs_start,
            im_end=rs_end,
            H=H,
            W=W,
            blur=blur,
            vg_path=vg_path,  # the time spectrum of rs is calculated relative to the vg time coordinate
            vg_start=rs_start,
            vg_end=rs_end + blur - 1
        )
        vg_info = dict(
            vg_path=vg_path,  # the time spectrum of rs is calculated relative to the vg time coordinate
            vg_start=rs_start,
            vg_end=rs_end + blur - 1,
            vg_bins=self.n_bins
        )
        return [rs_info], vg_info


class UnrollGevRSTestset(UnrollGevRSDataset):
    def __init__(self, opt):
        super(UnrollGevRSTestset, self).__init__(opt)
        self.data_root = opt['data_root']
        self.task = ['unroll']
        self.clear_only = self.opt['clear_only']  # whether to test with clear rs image
        self.n_tgt = 1
        self.rs_180_list = ['24209_1_30', '24209_1_31', '24209_1_33',
                            '24209_1_24', '24209_1_41', '24209_1_4']

        self.sample_meta = []  # meta information of each sample
        seq_dirs = [os.path.join(self.data_root, seq_name) for seq_name in os.listdir(self.data_root)]
        for seq_dir in tqdm(seq_dirs):
            self.sample_meta.extend(sorted(os.path.join(seq_dir, clip_name) for clip_name in os.listdir(seq_dir)))
        self.len = len(self.sample_meta)

    def choose_source(self, sample_path, clear_only=False):
        # select the rs to be identical to Gev-RS official
        seq_name = sample_path.split('/')[-2]
        if clear_only:
            if seq_name in self.rs_180_list:
                rs_impath_list = glob(os.path.join(sample_path, 'rs_000090_000269_1.png'))
            else:
                rs_impath_list = glob(os.path.join(sample_path, 'rs_000000_000359_1.png'))
        else:
            if seq_name in self.rs_180_list:
                rs_impath_list = glob(os.path.join(sample_path, 'rs_000090_000269_73.png'))
            else:
                rs_impath_list = glob(os.path.join(sample_path, 'rs_000000_000359_145.png'))

        rs_impath = random.choice(rs_impath_list)  # random select a rs frame # TODO: allow multiple RS frames
        W, H = Image.open(rs_impath).size
        rs_name = os.path.basename(rs_impath)
        im_type, rs_start, rs_end, blur = rs_name.strip('.png').split('_')
        rs_start, rs_end, blur = int(rs_start), int(rs_end), int(blur)
        vg_name = "vg_{}x{}_{}to{}_{}bins.npz".format(H, W, rs_start, rs_end+blur-1, self.n_bins)
        vg_path = os.path.join(sample_path, vg_name)  # path to the selected voxel grid

        rs_info = dict(
            im_type=im_type,
            im_path=rs_impath,
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
        im_info_example = im_info_list[0]
        seq_name = im_info_example['im_path'].split('/')[-4]
        if seq_name in self.rs_180_list:
            im_info = [im_info_list[90]]
        else:
            im_info = [im_info_list[180]]
        return im_info

    def __getitem__(self, index):
        data_dict = super(UnrollGevRSTestset, self).__getitem__(index)
        data_dict['sample_path'] = self.sample_meta[index]
        return data_dict


class UnrollGevRSDemoset(UnrollGevRSTestset):
    def __init__(self, opt, seq_dir=None):
        super(UnrollGevRSDemoset, self).__init__(opt)
        if seq_dir is not None:
            self.sample_meta = sorted([os.path.join(seq_dir, clip_name) for clip_name in os.listdir(seq_dir)])
        self.len = len(self.sample_meta)

    def choose_target(self, im_info_list):
        return im_info_list[::5]
