"""
Dataset for training Pixel Controlling
"""
import os
import math
from glob import glob
import random
from tqdm import tqdm

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms

import logging
logger = logging.getLogger('base')


def create_train_dataloader(dataset_opt, dataset_cls):
    # create dataset
    dataset = dataset_cls(dataset_opt)
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))

    # build data sampler
    if torch.distributed.is_initialized():
        from data.data_sampler import MTDistIterSampler
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        sampler = MTDistIterSampler(dataset, ratio=200)

        num_workers = dataset_opt['n_workers'] * world_size
        assert dataset_opt['batch_size'] % world_size == 0
        batch_size = dataset_opt['batch_size'] // world_size

    else:
        rank = -1
        num_workers = dataset_opt['n_workers']
        batch_size = dataset_opt['batch_size']
        sampler = None

    # get dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        num_workers=num_workers,
        sampler=sampler,
        drop_last=True,
        pin_memory=False
    )

    return dataloader, sampler


def find_available_vg(im_info_list, vg_file_list):
    """ the selected voxel grid should cover the source images """
    im_start_list = [im_info['im_start'] for im_info in im_info_list]
    im_end_list = [im_info['im_end'] + im_info['blur'] - 1 for im_info in im_info_list]
    all_expo_start = min(im_start_list)  # exposure start of the first frame of the clip
    all_expo_end = max(im_end_list)  # exposure end of the last frame of the clip

    available_vg_list = []
    for vg_file in vg_file_list:
        vg_name = os.path.basename(vg_file).strip('.npz')
        vg_type, res, ev_span, n_bins = vg_name.split('_')
        vg_start, vg_end = ev_span.split('to')
        vg_start, vg_end = int(vg_start), int(vg_end)
        if vg_start <= all_expo_start and vg_end >= all_expo_end:
            available_vg_list.append(vg_file)
    return available_vg_list


def get_reachable_gs(gs_list, vg_info):
    """ Get the gt candidates that can be covered by the vg span """
    gs_list = gs_list[vg_info['vg_start']:(vg_info['vg_end'] + 1)]  # list of paths to reachable gs frames
    gs_offset = list(range(vg_info['vg_start'], vg_info['vg_end'] + 1))  # list of offsets-in-clip of the gs frames
    gs_info_list = []
    for gs_path, offset in zip(gs_list, gs_offset):
        W, H = Image.open(gs_path).size
        gs_info = dict(
            im_type='gs',
            im_path=gs_path,
            im_start=offset,
            im_end=offset,
            H=H,
            W=W,
            blur=1,
            vg_path=vg_info['vg_path'],  # the time spectrum of rs is calculated relative to the vg time coordinate
            vg_start=vg_info['vg_start'],
            vg_end=vg_info['vg_end']
        )
        gs_info_list.append(gs_info)
    return gs_info_list


class FECubeDataset(data.Dataset):
    """ loading data for multiple tasks """
    def __init__(self, opt):
        super(FECubeDataset, self).__init__()
        self.opt = opt
        self.data_root = opt['data_root']

        # data parameters
        self.n_tgt = opt['n_tgt']  # number of predictions per INR
        self.n_bins = opt['n_bins']
        self.im_with_time = self.opt['im_with_time']
        self.clear_only = self.opt['clear_only']

        # augmentation options
        self.crop_size = opt['crop_size']
        self.flip_ratio = opt['flip_ratio']
        self.rot_ratio = opt['rot_ratio']

        # arrange data from directory tree
        self.build_data_tree()

    def build_data_tree(self):
        self.sample_meta = []  # meta information of each sample
        if isinstance(self.data_root, str):
           self.data_root = [self.data_root]

        # collect meta data
        for data_root in self.data_root:
            seq_dirs = [os.path.join(data_root, seq_name) for seq_name in os.listdir(data_root)]
            seq_dirs = [t for t in seq_dirs if os.path.isdir(t)]  # ignore non-folder items
            for seq_dir in tqdm(seq_dirs):
                self.sample_meta.extend([os.path.join(seq_dir, clip_name) for clip_name in os.listdir(seq_dir)])

        self.len = len(self.sample_meta)
        self.sample_info_buffer = dict(
            gs_list=dict(),
            ev_npz_list=dict(),
        )

    def load_images(self, im_info_list, load_with_time=True):
        """
        Args:
            im_info_list: list of image paths
            load_with_time: whether to load the time map together with the image
        """
        im_list = []
        tspec_list = []
        for im_info in im_info_list:
            im_path = im_info['im_path']
            im = np.array(Image.open(im_path).convert('RGB')).astype(np.float32) / 255
            im_list.append(im)

            if load_with_time:
                im_tspec = self.get_tspec(im_info)
                tspec_list.append(im_tspec)

        return im_list, tspec_list

    def get_tspec(self, im_info):
        """ We calculate the time in the unit of frame
        e.g. the im_start is 0-frame, im_end is 179-frame
        If the frame rate of video is 1000 fps, this rs span will be equivalent to 179ms,
        The time spectrum is calculated relative to the voxel grid span,
        i.e. (tspec_a - vg_start) / vg_span
        """
        im_H, im_W = im_info['H'], im_info['W']
        im_start = im_info['im_start']
        im_end = im_info['im_end']
        blur = im_info['blur']
        vg_start = im_info['vg_start']
        vg_end = im_info['vg_end']
        vg_span = vg_end - vg_start

        tspec_a = np.concatenate([np.linspace(im_start, im_end, im_H)[:, np.newaxis]] * im_W, axis=1).astype(np.float)[..., None]
        tspec_b = np.concatenate([np.linspace(im_start+blur-1, im_end+blur-1, im_H)[:, np.newaxis]] * im_W, axis=1).astype(np.float)[..., None]
        tspec = np.concatenate([tspec_a, tspec_b], axis=-1)
        tspec = (tspec - vg_start) / vg_span  # normalized with vg
        # cv2.imwrite('a.png', (tspec[:, :, 0] * 255).astype(np.uint8))  # for debug
        # cv2.imwrite('b.png', (tspec[:, :, 1] * 255).astype(np.uint8))  # for debug
        assert (tspec >= 0).all() and (tspec <= 1).all()

        return tspec

    def get_gs_list(self, sample_path):
        """ Buffer the gt candidates of previously loaded sample """
        gs_list = self.sample_info_buffer['gs_list'].get(sample_path, None)
        if gs_list is None:
            gs_list = sorted(glob(os.path.join(sample_path, 'gs/*.png')))
            self.sample_info_buffer['gs_list'][sample_path] = gs_list
        return gs_list

    def choose_target(self, im_info_list):
        return random.sample(im_info_list, k=self.n_tgt)

    def choose_source(self, sample_path, clear_only=False):
        """ choose the source rgb image (abstract method)
        For deblur task, a blurry image will be selected
        For VFI, a pair of sharp keyframes will be selected
        For JDFI, a pair of blurry keyframes will be selected
        For Unrolling, an RS image will be selected
        etc.
        """
        raise NotImplementedError

    def data_augmentation(self, data):
        """ data augmentation for improve training """

        # data items registry
        src_img = data['src_img']
        src_tspec = data['src_tspec']
        vg = data['vg']
        vg_tspec = data['vg_tspec']
        target_gs_imgs = data['target_gs_imgs']
        tgt_tspec = data['tgt_tspec']

        _, H, W, C = src_img.shape

        # RandCrop
        if self.crop_size < 0:
            pass
        else:
            crop_size = self.crop_size
            rnd_h = random.randint(0, max(0, H - crop_size))  # random topleft
            rnd_w = random.randint(0, max(0, W - crop_size))  # random topleft
            # crop voxel grid
            vg = vg[:, rnd_h:rnd_h + crop_size, rnd_w:rnd_w + crop_size]
            # crop vg timespec
            vg_tspec = vg_tspec[:, rnd_h:rnd_h + crop_size, rnd_w:rnd_w + crop_size]
            # crop source image
            src_img = src_img[:, rnd_h:rnd_h + crop_size, rnd_w:rnd_w + crop_size, :]
            # crop source timespec
            src_tspec = src_tspec[:, rnd_h:rnd_h + crop_size, rnd_w:rnd_w + crop_size, :]
            # crop gt image
            target_gs_imgs = target_gs_imgs[:, rnd_h:rnd_h + crop_size, rnd_w:rnd_w + crop_size, :]
            # crop target timespec
            tgt_tspec = tgt_tspec[:, rnd_h:rnd_h + crop_size, rnd_w:rnd_w + crop_size, :]

        # ToTensor
        vg = torch.from_numpy(vg)[None].float()
        vg_tspec = torch.from_numpy(vg_tspec)[None].float()
        src_img = torch.from_numpy(src_img).permute(0, 3, 1, 2).float()  # [B, N, C, H, W]
        src_tspec = torch.from_numpy(src_tspec).permute(0, 3, 1, 2).float()  # [B, N, C, H, W]
        target_gs_imgs = torch.from_numpy(target_gs_imgs).permute(0, 3, 1, 2).float()  # [B, T, C, H, W]
        tgt_tspec = torch.from_numpy(tgt_tspec).permute(0, 3, 1, 2).float()  # [B, T, C, H, W]

        # Rand Rotate
        if random.uniform(0, 1) < self.rot_ratio:
            rot_func = transforms.RandomRotation([90, 90]) if random.uniform(0, 1) <= 0.5 else transforms.RandomRotation([-90, -90])
            vg = rot_func(vg)
            vg_tspec = rot_func(vg_tspec)
            src_img = rot_func(src_img)
            src_tspec = rot_func(src_tspec)
            target_gs_imgs = rot_func(target_gs_imgs)
            tgt_tspec = rot_func(tgt_tspec)

        # RandFlip
        if random.uniform(0, 1) < self.flip_ratio:  # RandomHorizontalFlip
            flip_func = transforms.RandomHorizontalFlip(p=1)
            vg = flip_func(vg)
            vg_tspec = flip_func(vg_tspec)
            src_img = flip_func(src_img)
            src_tspec = flip_func(src_tspec)
            target_gs_imgs = flip_func(target_gs_imgs)
            tgt_tspec = flip_func(tgt_tspec)

        if random.uniform(0, 1) < self.flip_ratio:  # RandomVerticalFlip
            flip_func = transforms.RandomVerticalFlip(p=1)
            vg = flip_func(vg)
            vg_tspec = flip_func(vg_tspec)
            src_img = flip_func(src_img)
            src_tspec = flip_func(src_tspec)
            target_gs_imgs = flip_func(target_gs_imgs)
            tgt_tspec = flip_func(tgt_tspec)

        # TODO: Add Noise
        # TODO: Color Distortion

        data = {
            'src_img': src_img,
            'src_tspec': src_tspec,
            'vg': vg,
            'vg_tspec': vg_tspec,
            'target_gs_imgs': target_gs_imgs,
            'tgt_tspec': tgt_tspec,
        }
        return data

    def get_data(self, index):
        sample_path = self.sample_meta[index]  # path to the sample clip, where exists all kinds of images and vg
        src_iminfo_list, vg_info = self.choose_source(sample_path, clear_only=self.clear_only)

        # Load Source Images
        src_img, src_tspec = self.load_images(src_iminfo_list, self.im_with_time)
        src_img = np.stack(src_img, axis=0)
        src_tspec = np.stack(src_tspec, axis=0)
        _, H, W, C = src_img.shape

        # load voxel grid
        vg = np.load(vg_info['vg_path'])['arr_0']
        vg_tspec = np.ones(vg.shape[1:])[None, :, :] * np.linspace(0, 1, vg.shape[0] + 1)[:, None, None]

        # Load Target Images
        '''
        When H is 3, the data is as follows:
            GS imgae:       0.png -------------- 1.png ------------- 2.png
            RS image:          | --------------- rs.png -------------- |
            Event:             | ------1.npz------ | ------2.npz------ |
            Time Stamp:       0.0              1/(H-1)=0.5         1/(H-1)=1.0 
        We take rs.png as input, we can use [0.png, 1.png, 2.png] as GT, 
        whose corresponding time stamps are [ 0.0 , 1/(H-1)=0.5, 2/(H-1)=1.0]  
        '''
        gs_list = self.get_gs_list(sample_path)  # get all gs frames in the clip
        gs_info_list = get_reachable_gs(gs_list, vg_info)  # pick the frames covered by vg, collect useful info
        gs_info_list = self.choose_target(gs_info_list)
        target_gs_imgs, tgt_tspec = self.load_images(gs_info_list, load_with_time=True)
        target_gs_imgs = np.stack(target_gs_imgs, axis=0)
        tgt_tspec = np.stack(tgt_tspec, axis=0)

        data = {
            'src_img': src_img,
            'src_tspec': src_tspec,
            'vg': vg,
            'vg_tspec': vg_tspec,
            'target_gs_imgs': target_gs_imgs,
            'tgt_tspec': tgt_tspec,
        }

        data = self.data_augmentation(data)

        # data items unregistry
        src_img = data['src_img']
        src_tspec = data['src_tspec']
        vg = data['vg']
        vg_tspec = data['vg_tspec']
        target_gs_imgs = data['target_gs_imgs']
        tgt_tspec = data['tgt_tspec']

        # Collection
        return {
            'src_img': src_img,  # source images
            'src_tspec': src_tspec,  # time spectrum of source images
            'vg': vg.squeeze(0),  # voxel grid loaded
            'vg_tspec': vg_tspec.squeeze(0),  # time spectrum of voxel grid
            'target_gs_imgs': target_gs_imgs,  # the target image, GT for training
            'tgt_tspec': tgt_tspec,
            # the time spectrum of the target image, which tell the network which pixel is wanted
        }

    def __getitem__(self, index):
        n_try = 0
        while True:
            n_try += 1
            if n_try > 10:
                raise RuntimeError(f"Fail to load data after {n_try} times attempt")

            try:
                data = self.get_data(index)
                return data
            except:
                print("error of loading data, skipped")
                index = index + 1
                continue

    def __len__(self):
        return self.len
