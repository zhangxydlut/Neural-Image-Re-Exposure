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

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)


def read_image(img_path):
    '''read one image from img_path
    Return img: HWC, BGR, [0,1], numpy
    '''
    img_GT = cv2.imread(img_path)
    img = img_GT.astype(np.float32) / 255.
    return img


def read_seq_imgs(img_seq_path):
    '''read a sequence of images'''
    img_path_l = glob.glob(img_seq_path + '/*')
    # img_path_l.sort(key=lambda x: int(os.path.basename(x)[:-4]))
    img_path_l.sort(key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
    img_l = [read_image(v) for v in img_path_l]
    # stack to TCHW, RGB, [0,1], torch
    imgs = np.stack(img_l, axis=0)
    imgs = imgs[:, :, :, [2, 1, 0]]
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
    return imgs


def test_index_generation(skip, N_out, len_in):
    '''
    params:
    skip: if skip even number;
    N_out: number of frames of the network;
    len_in: length of input frames
    example:
  len_in | N_out  | times | (no skip)                  |   (skip)
    5    |   3    |  4/2  | [0,1], [1,2], [2,3], [3,4] | [0,2],[2,4]
    7    |   3    |  5/3  | [0,1],[1,2][2,3]...[5,6]   | [0,2],[2,4],[4,6]
    5    |   5    |  2/1  | [0,1,2] [2,3,4]            | [0,2,4]
    '''
    # number of input frames for the network
    N_in = 1 + N_out // 2
    # input length should be enough to generate the output frames
    assert N_in <= len_in

    sele_list = []
    if skip:
        right = N_out  # init
        while (right <= len_in):
            h_list = [right - N_out + x for x in range(N_out)]
            l_list = h_list[::2]
            right += (N_out - 1)
            sele_list.append([l_list, h_list])
    else:
        right = N_out  # init
        right_in = N_in
        while (right_in <= len_in):
            h_list = [right - N_out + x for x in range(N_out)]
            l_list = [right_in - N_in + x for x in range(N_in)]
            right += (N_out - 1)
            right_in += (N_in - 1)
            sele_list.append([l_list, h_list])
    # check if it covers the last image, if not, we should cover it
    if (skip) and (right < len_in - 1):
        h_list = [len_in - N_out + x for x in range(N_out)]
        l_list = h_list[::2]
        sele_list.append([l_list, h_list])
    if (not skip) and (right_in < len_in - 1):
        right = len_in * 2 - 1;
        h_list = [right - N_out + x for x in range(N_out)]
        l_list = [len_in - N_in + x for x in range(N_in)]
        sele_list.append([l_list, h_list])
    return sele_list


####################
# video
####################

def extract_frames(ffmpeg_dir, video, outDir):
    """
    Converts the `video` to images.
    Parameters
    ----------
        video : string
            full path to the video file.
        outDir : string
            path to directory to output the extracted images.
    Returns
    -------
        error : string
            Error message if error occurs otherwise blank string.
    """

    error = ""
    print('{} -i {} -vsync 0 {}/%06d.png'.format(os.path.join(ffmpeg_dir, "ffmpeg"), video, outDir))
    retn = os.system('{} -i "{}" -vsync 0 {}/%06d.png'.format(os.path.join(ffmpeg_dir, "ffmpeg"), video, outDir))
    if retn:
        error = "Error converting file:{}. Exiting.".format(video)
    return error


def create_video(ffmpeg_dir, dir, output, fps):
    error = ""
    # print('{} -r {} -i {}/%6d.png -vcodec ffvhuff {}'.format(os.path.join(ffmpeg_dir, "ffmpeg"), fps, dir, output))
    # retn = os.system('{} -r {} -i {}/%6d.png -vcodec ffvhuff "{}"'.format(os.path.join(ffmpeg_dir, "ffmpeg"), fps, dir, output))
    print('{} -r {} -f image2 -i {}/%6d.png {}'.format(os.path.join(ffmpeg_dir, "ffmpeg"), fps, dir, output))
    retn = os.system('{} -r {} -f image2 -i {}/%6d.png {}'.format(os.path.join(ffmpeg_dir, "ffmpeg"), fps, dir, output))
    if retn:
        error = "Error creating output video. Exiting."
    return error
