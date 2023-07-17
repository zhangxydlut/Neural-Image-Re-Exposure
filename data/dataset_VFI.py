"""
Dataset for VFI and JDFI
"""
import os
import math
from glob import glob
import random
from tqdm import tqdm
from PIL import Image

from data.dataset_FECube import FECubeDataset, find_available_vg, create_train_dataloader

import logging
logger = logging.getLogger('base')


class VFIGoProDataset(FECubeDataset):
    def choose_source(self, sample_path, clear_only=False):
        impath_list = glob(os.path.join(sample_path, 'gs_*.png'))  # allow blurry src image, compatible with jdfi

        vg_file_list = glob(os.path.join(sample_path, 'vg_*{}bins.npz'.format(self.n_bins)))

        im_info_list = []
        for i in range(2):
            im_path = random.choice(impath_list)  # random select a rs frame
            W, H = Image.open(im_path).size
            im_name = os.path.basename(im_path)
            im_type, rs_start, rs_end, blur = im_name.replace('.png', '').split('_')
            rs_start, rs_end, blur = int(rs_start), int(rs_end), int(blur)

            im_info = dict(
                im_type=im_type,
                im_path=im_path,
                im_start=rs_start,
                im_end=rs_end,
                blur=blur,
                H=H,
                W=W,
            )
            im_info_list.append(im_info)

        available_vg_list = find_available_vg(im_info_list, vg_file_list)  # select the vg cover the src images
        vg_path = random.choice(available_vg_list)
        vg_start, vg_end = os.path.basename(vg_path).strip('.npz').split('_')[2].split('to')
        vg_start, vg_end = int(vg_start), int(vg_end)

        for im_info in im_info_list:
            im_info['vg_path'] = vg_path
            im_info['vg_start'] = vg_start
            im_info['vg_end'] = vg_end

        vg_info = dict(
            vg_path=vg_path,  # the time spectrum of rs is calculated relative to the vg time coordinate
            vg_start=vg_start,
            vg_end=vg_end,
            vg_bins=self.n_bins
        )
        return im_info_list, vg_info


class VFIGoProTestset(VFIGoProDataset):
    def __init__(self, opt):
        super(VFIGoProTestset, self).__init__(opt)
        self.data_root = opt['data_root']
        self.task = ['vfi']
        self.clear_only = opt['clear_only']  # whether to test with clear rs image
        self.tgt_indices = opt['tgt_indices']

        self.sample_meta = []  # meta information of each sample
        seq_dirs = [os.path.join(self.data_root, seq_name) for seq_name in os.listdir(self.data_root)]
        for seq_dir in tqdm(seq_dirs):
            self.sample_meta.extend(sorted(os.path.join(seq_dir, clip_name) for clip_name in os.listdir(seq_dir)))
        self.len = len(self.sample_meta)

    def choose_source(self, sample_path, clear_only=False):
        impath_list = sorted(glob(os.path.join(sample_path, 'gs_*.png')))  # ordered src
        assert len(impath_list) == 2  # only support VFI with 2 source frame
        vg_file_list = glob(os.path.join(sample_path, 'vg_*{}bins.npz'.format(self.n_bins)))

        im_info_list = []
        for i in range(2):
            im_path = impath_list[i]  # ordered src
            W, H = Image.open(im_path).size
            im_name = os.path.basename(im_path)
            im_type, rs_start, rs_end, blur = im_name.replace('.png', '').split('_')
            rs_start, rs_end, blur = int(rs_start), int(rs_end), int(blur)

            im_info = dict(
                im_type=im_type,
                im_path=im_path,
                im_start=rs_start,
                im_end=rs_end,
                blur=blur,
                H=H,
                W=W,
            )
            im_info_list.append(im_info)

        available_vg_list = find_available_vg(im_info_list, vg_file_list)  # select the vg cover the src images
        # vg_path = random.choice(available_vg_list)
        assert len(available_vg_list) == 1  # There should be only one vg for the test set, which contains events inbetween the two src frame
        vg_path = available_vg_list[0]
        vg_start, vg_end = os.path.basename(vg_path).strip('.npz').split('_')[2].split('to')
        vg_start, vg_end = int(vg_start), int(vg_end)

        for im_info in im_info_list:
            im_info['vg_path'] = vg_path
            im_info['vg_start'] = vg_start
            im_info['vg_end'] = vg_end

        vg_info = dict(
            vg_path=vg_path,  # the time spectrum of rs is calculated relative to the vg time coordinate
            vg_start=vg_start,
            vg_end=vg_end,
            vg_bins=self.n_bins
        )
        return im_info_list, vg_info

    def choose_target(self, im_info_list):
        return [im_info_list[t] for t in self.tgt_indices]

    def __getitem__(self, index):
        data_dict = super(VFIGoProTestset, self).__getitem__(index)

        # # crop_region = (320, 180, 960, 540)
        # # crop_region = (0, 0, 1280, 720)
        # crop_region = (0, 0, 256, 256)
        # x1, y1, x2, y2, = crop_region
        # data_dict['src_img'] = data_dict['src_img'][:, :, y1:y2, x1:x2]
        # data_dict['src_tspec'] = data_dict['src_tspec'][:, :, y1:y2, x1:x2]
        # data_dict['vg'] = data_dict['vg'][:, y1:y2, x1:x2]
        # data_dict['vg_tspec'] = data_dict['vg_tspec'][:, y1:y2, x1:x2]
        # data_dict['target_gs_imgs'] = data_dict['target_gs_imgs'][:, :, y1:y2, x1:x2]
        # data_dict['tgt_tspec'] = data_dict['tgt_tspec'][:, :, y1:y2, x1:x2]

        data_dict['sample_path'] = self.sample_meta[index]
        data_dict['sample_id'] = '{}-{}'.format(*data_dict['sample_path'].split('/')[-2:])
        return data_dict

