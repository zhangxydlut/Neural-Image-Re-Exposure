"""
Visualize the voxelized events in the generated Samples
- vis_rgbe: superpose events to the rgb image (RGBE visualization)
- reconstruct_seq: visualize by V2E

Usage:
    After running `data/tools/scripts/data_process_gevrs.py`
    ```
    # check all sequences
    python sanity_samples.py

    # check specified sequence
    python sanity_check.py --source_root=./Processed/test/Sample/24209_1_33
    ```
"""
import os
import glob
import numpy as np
import cv2
import sys
import argparse

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from tools.visualization.visualize_ERGB import ERGB_Player

VG_NAME = 'vg_720x1280_0to10_36bins.npz'
RS_NAME = 'gs_000000_000000_11.png'


def parse_args():
    """ Please keep all the parameters as default """
    parser = argparse.ArgumentParser(description='parameters for visualize V2E result')
    parser.add_argument('--source_root', default=None, type=str, help='path to folder of V2E output')
    parser.add_argument('--width', default=1280, type=int, help='the width of image')
    parser.add_argument('--height', default=720, type=int, help='the height of image')
    args = parser.parse_args()
    return args


def check_clip_e2vid(seq_root, W, H, vis=True, save=False):
    """ load the raw events in the clip and perform E2VID """
    E2VBatch = 50  # 60
    from tools.rpg_e2vid.e2vid import build_e2vid_reconstructor

    def checkdir(dir):
        if not os.path.exists(dir):
            print("creating {} ...".format(dir))
            os.makedirs(dir)
        return dir

    def get_rgbe_pair(clip_dir):
        """ iterate the event files and image files """
        ev_npys = sorted(glob.glob(os.path.join(clip_dir, 'event/*.npz')))
        frames = sorted(glob.glob(os.path.join(clip_dir, 'gs/*.png')))
        for idx, (ev_npy, frame) in enumerate(zip(ev_npys, frames)):
            ev_pack = np.array(np.load(ev_npy)['arr_0'])
            # TODO summary multiple ev_packs
            image = cv2.imread(frame)
            yield image, ev_pack

    reconstructor = build_e2vid_reconstructor(width=W, height=H)
    clip_dir = os.path.join(seq_root)
    if save:
        e2vid_dir = checkdir(os.path.join(seq_root, 'vis/e2vid'))
    ev_packs = []
    for idx, (image, ev_pack) in enumerate(get_rgbe_pair(clip_dir)):
        ev_pack = np.array(ev_pack)
        ev_pack = np.stack([ev_pack['timestamp'], ev_pack['x'], ev_pack['y'], ev_pack['polarity']], axis=1).astype(np.float64)
        ev_packs.append(ev_pack)
        if not (idx+1) % E2VBatch == 0:  # accumulate 60 npz for visualization
            continue
        ev_pack = np.concatenate(ev_packs, axis=0)
        e2vid = reconstructor.update(ev_pack)
        ev_packs = []
        if vis:
            cv2.namedWindow('e2vid')
            cv2.resizeWindow('e2vid', width=640, height=360)
            cv2.imshow('e2vid', e2vid); cv2.waitKey(0)
        if save:
            e2vid_file = os.path.join(e2vid_dir, 'e2vid_{:06}.png'.format(idx))
            print("saving {}".format(e2vid_file))
            cv2.imwrite(e2vid_file, e2vid)


def check_clip_vg(seq_root, W, H, vis=True, save=False):
    """ load the preprocessed voxel grid in the clip and perform E2VID """
    from tools.rpg_e2vid.e2vid import build_e2vid_reconstructor

    def checkdir(dir):
        if not os.path.exists(dir):
            print("creating {} ...".format(dir))
            os.makedirs(dir)
        return dir

    def get_rgbe_pair(clip_dir):
        """ iterate the event files and image files """
        vg_path = os.path.join(clip_dir, VG_NAME)
        rs_frame = os.path.join(clip_dir, RS_NAME)
        vg = np.load(vg_path)['arr_0']
        for idx in range(int(np.floor(vg.shape[0] / 5))):
            yield rs_frame, vg[5*idx: 5*(idx+1)]

    reconstructor = build_e2vid_reconstructor(width=W, height=H)
    clip_dir = os.path.join(seq_root)
    for idx, (image, vg) in enumerate(get_rgbe_pair(clip_dir)):
        e2vid = reconstructor.constructor.update(torch.from_numpy(vg).float().cuda())
        if vis:
            cv2.namedWindow('e2vid')
            cv2.resizeWindow('e2vid', width=640, height=360)
            cv2.imshow('e2vid', e2vid); cv2.waitKey(0)
        if save:
            e2vid_dir = checkdir(os.path.join(seq_root, 'vis/e2vid'))
            e2vid_file = os.path.join(e2vid_dir, 'e2vid_{:06}.png'.format(idx))
            print("saving {}".format(e2vid_file))
            cv2.imwrite(e2vid_file, e2vid)


def check_clip_rgbe(seq_root, vis=True, save=False):
    RGBEBatch = 5
    player = ERGB_Player()

    def get_e2vout_events(ev_dir):
        event_files = sorted(glob.glob(os.path.join(ev_dir, '*.npz')))
        for evf in event_files:
            evs = np.load(evf)['arr_0']
            yield evs

    def get_e2vout_images(img_dir):
        image_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        for im_path in image_files:
            im = cv2.imread(im_path)
            yield im

    print(seq_root)
    events = get_e2vout_events(os.path.join(seq_root, 'event'))
    images = get_e2vout_images(os.path.join(seq_root, 'gs'))

    event_buffer = []
    for frame_idx, (img, evs) in enumerate(zip(images, events)):
        if not (frame_idx + 1) % RGBEBatch == 0:  # accumulate for 5 npz for showing
            event_buffer.append(evs)
            continue
        evs = np.concatenate(event_buffer)
        event_buffer = []
        n_evs = len(evs['polarity'])
        evs_split = 1
        for i in range(evs_split):
            a = n_evs // evs_split * i
            b = n_evs // evs_split * (i+1)
            _evs = evs[a:b]
            ergb_spp = player.ergb_superposed_show(img, _evs, vis=vis, win_size=(args.height, args.width))

            k = player.waitKey(0)
            if save:
                target_dir = os.path.join(seq_root, 'vis/ergb_vis')
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                cv2.imwrite(os.path.join(target_dir, 'ergb_{:05}.png'.format(frame_idx)), ergb_spp)


def check_rs_with_event(seq_root, vis=True, save=False):
    RGBEBatch = 2
    player = ERGB_Player()

    def get_e2vout_events(ev_dir):
        vg_path = os.path.join(clip_dir, VG_NAME)
        a, b = VG_NAME.split('_')[2].split('to')
        a, b = int(a), int(b)
        event_files = sorted(glob.glob(os.path.join(ev_dir, '*.npz')))[a:b]  # TODO: get event according to VG_NAME
        for evf in event_files:
            evs = np.load(evf)['arr_0']
            yield evs

    def get_e2vout_images(rs_frame):
        while True:
            im = cv2.imread(rs_frame)
            yield im

    print(seq_root)
    events = get_e2vout_events(os.path.join(seq_root, 'event'))
    images = get_e2vout_images(os.path.join(seq_root, RS_NAME))

    event_buffer = []
    for frame_idx, (img, evs) in enumerate(zip(images, events)):
        if not (frame_idx + 1) % RGBEBatch == 0:  # accumulate for 5 npz for showing
            event_buffer.append(evs)
            continue
        evs = np.concatenate(event_buffer)
        event_buffer = []
        n_evs = len(evs['polarity'])
        evs_split = 1
        for i in range(evs_split):
            a = n_evs // evs_split * i
            b = n_evs // evs_split * (i+1)
            _evs = evs[a:b]
            ergb_spp = player.ergb_superposed_show(img, _evs, vis=vis, win_size=(args.height, args.width))

            k = player.waitKey(0)
            if save:
                target_dir = os.path.join(seq_root, 'vis/ergb_vis')
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                cv2.imwrite(os.path.join(target_dir, 'ergb_{:05}.png'.format(frame_idx)), ergb_spp)


def sanity_check_vg_completion():
    """ Check if all generated voxel grid is completed """
    for vg_path in tqdm.tqdm(sorted(glob.glob('./Processed/train/Sample/*/*/*npz'))):
        try:
            np.load(vg_path)['arr_0']
        except:
            print("*(***************")
            print(vg_path)
            print("*(***************")
            pdb.set_trace()

    for vg_path in tqdm.tqdm(sorted(glob.glob('./Processed/test/Sample/*/*/*npz'))):
        try:
            np.load(vg_path)['arr_0']
        except:
            print("*(***************")
            print(vg_path)
            print("*(***************")
            pdb.set_trace()


if __name__ == '__main__':
    args = parse_args()
    if args.source_root is None:
        SAMPLE_ROOT = './Processed/test/Sample'  # SEQ_ROOT = './Processed/test/Sample/24209_1_33'
        seq_dir_list = [os.path.join(SAMPLE_ROOT, seq_name) for seq_name in os.listdir(SAMPLE_ROOT)]
    else:
        seq_dir_list = [args.source_root]

    for seq_dir in seq_dir_list:
        for clip_dir in [os.path.join(seq_dir, clip_name) for clip_name in sorted(os.listdir(seq_dir))]:
            # check specified sample
            check_clip_vg(seq_root=clip_dir, W=args.width, H=args.height, vis=True, save=False)
            check_rs_with_event(seq_root=clip_dir, vis=True, save=False)

            # check data of whole clip
            check_clip_e2vid(seq_root=clip_dir, W=args.width, H=args.height, vis=True, save=False)
            check_clip_rgbe(seq_root=clip_dir, vis=True, save=False)
