"""
VFI and BIN Data preprocessing script of GoPro.
References:
    In https://github.com/JihyongOh/DeMFI#quick-start-for-evaluations-on-test-datasets-deblurring-and-multi-frame-interpolation-x8-as-in-table-2
    ```
    We follow a blurry formation setting from BIN (Blurry Video Frame Interpolation)
    by averaging 11 consecutive frames at a stride of 8 frames over time to synthesize
    blurry frames captured by a long exposure, which finally generates blurry frames of
    30fps with K = 8 and τ = 5 in Eq. 1.
    ```

Usage:
1. Install V2E (Please refer to eventinr/data/tools/v2e/README.md)
2. Put raw video into DATA_ROOT/RAW_DIR
3. python data_process_gopro.py

Validation:
    run `python sanity_check.py`
"""
import os
import glob
import cv2
import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
import random
from data_process_voxelize import generate_voxel_grid
random.seed(123)

W, H = (1280, 720)
split = 'train'
# Paths
DATA_ROOT = os.path.abspath(f"./JDFI")
# DATA_ROOT = os.path.abspath(f"/media/zxy/AnyPixEF/Dataset/GoPro1280x720/VFI")
RAW_DIR = os.path.join(DATA_ROOT, f"../VideoRaw/{split}")
PROCESSED_DIR = os.path.join(DATA_ROOT, f"Processed/{split}")
RS_DIR = os.path.join(PROCESSED_DIR, 'RS')
GS_DIR = os.path.join(PROCESSED_DIR, 'GS')
EVENT_DIR = os.path.join(PROCESSED_DIR, 'Event')
SAMPLE_DIR = os.path.join(PROCESSED_DIR, 'Sample')

# Hyper parameters
N_BINS = 36
CLIP_STEP = 8
blur_candidates = [11, 1]
CLIP_LEN = CLIP_STEP + max(blur_candidates)


def makedir(dir):
    if not os.path.exists(dir):
        print(f"{dir} not exists! Creating...")
        os.makedirs(dir)
    return dir


def imshow(img, win_name='', wait=0):
    cv2.imshow(win_name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # from CV-RGB to PIL-RGB
    cv2.waitKey(wait)


def imwrite(save_path, img):
    if not os.path.exists(os.path.dirname(save_path)):
        print(f"Creating directory {os.path.dirname(save_path)}")
        os.makedirs(os.path.dirname(save_path))
    iio.imwrite(save_path, img)
    print("{} saved!".format(save_path))


def get_gs_clip(vid, start_index, clip_length, im_size=(1280, 720)):
    """
    Args:
        vid: path to the video
        im_size: size of image
    Returns:
        list of RGB frames
    """
    W, H = im_size

    frame_list = [os.path.join(vid, im_name) for im_name in sorted(os.listdir(vid))]
    clip = frame_list[start_index:start_index+clip_length]
    clip = [np.array(Image.fromarray(iio.imread(im)).resize((W, H))) for im in clip]
    return clip


def get_event(vid, event_dir, im_size, save_vid=False):
    W, H = im_size

    # ====================== V2E Parameters =============================
    # Camera Specific
    POS_THRES = 0.20  # default 0.2
    NEG_THRES = 0.20  # default 0.2
    SIGMA_THRES = 0.03  # default 0.03
    CUTOFF_HZ = 300  # default 300; 300 is recommended by Prof. Shi

    # Data Specific
    EV_SPACE_W = W  # 346 for DAVIS346; 1280 for Prophesee
    EV_SPACE_H = H  # 260 for DAVIS346; 720 for Prophesee
    STOP_TIME = -1  # take ${STOP_TIME} seconds video
    TIMESTAMP_RESOLUTION = .0001  # atomic time unit for events, use 1e-4 as default
    INPUT_FRAME_RATE = 250  # TODO: this doesn't matter because the normalized timestamp will be used in the program

    # IO
    INPUT = vid
    OUTPUT = event_dir

    old_cwd = os.getcwd()
    os.chdir('../../tools/v2e')
    cmd = f"python v2e.py -i  {INPUT} " \
          f"--input_frame_rate={INPUT_FRAME_RATE} " \
          f"--output_folder={OUTPUT} " \
          f"--batch_size=4 " \
          f"--overwrite " \
          f"--no_preview " \
          f"--output_width={EV_SPACE_W} " \
          f"--output_height={EV_SPACE_H} " \
          f"--auto_timestamp_resolution=False " \
          f"--dvs_exposure duration 0.005 " \
          f"--pos_thres={POS_THRES} " \
          f"--neg_thres={NEG_THRES} " \
          f"--sigma_thres={SIGMA_THRES} " \
          f"--stop_time={STOP_TIME} " \
          f"--cutoff_hz={CUTOFF_HZ} " \
          f"--timestamp_resolution={TIMESTAMP_RESOLUTION} "

    if save_vid:
        cmd += f"--vid_orig=video_orig.avi "
        cmd += f"--vid_slomo=video_slomo.avi "

    os.system(cmd)
    os.chdir(old_cwd)
    return event_dir


def run_gopro_process():
    """ formulate the dataset into desired folder structure """
    vid_path_list = glob.glob(os.path.join(RAW_DIR, '*'))

    for vid in vid_path_list:
        torch.cuda.empty_cache()
        vid_name = os.path.basename(vid)
        clip_start = 0

        # if vid_name not in ['GOPR0854_11_00']:  # process specified sequences only | GOPR0374_11_02 GOPR0854_11_00
        #     continue

        # if os.path.exists(os.path.join(SAMPLE_DIR, vid_name)):  # skip previously processed
        #     print("{} already exists".format(vid_name))
        #     continue

        if not os.path.exists(os.path.join(EVENT_DIR, vid_name)):  # skip processed event
            get_event(vid, event_dir=os.path.join(EVENT_DIR, vid_name), im_size=(W, H))

        # synthesize rs frames
        while True:
            sample_dir = os.path.join(SAMPLE_DIR, vid_name, "{:06}_{:06}".format(clip_start, clip_start+1+CLIP_LEN-1))

            clip = get_gs_clip(vid, start_index=clip_start, clip_length=1+CLIP_LEN-1, im_size=(W, H))
            if len(clip) < CLIP_LEN or (clip_start + CLIP_LEN) > len(os.listdir(os.path.join(EVENT_DIR, vid_name, 'ev_npz'))):
                break

            # save gs frames (as GT) and raw synthesized event
            sample_gs_dir = makedir(os.path.join(sample_dir, 'gs'))
            sample_event_dir = makedir(os.path.join(sample_dir, 'event'))

            # collect and link the gs frames and events to the `sample directory`
            for _offset, frame in enumerate(clip):
                gs_save_path = os.path.join(GS_DIR, "{}/{:06}.png".format(vid_name, clip_start + _offset))
                if not os.path.exists(gs_save_path):
                    imwrite(gs_save_path, frame)
                else:
                    print(f"{gs_save_path} exists!")
                    t = iio.imread(gs_save_path)
                    assert ((frame - t) == 0).all()
                # link gs frames to corresponding sample dir
                gs_save_path_rel = gs_save_path.replace(PROCESSED_DIR, '../../../..')  # relative path
                os.system(f"ln -s {gs_save_path_rel} {sample_gs_dir}")

                # link event npz to corresponding sample dir
                '''
                0.png -------------- 1.png ------------- 2.png
                   | ------1.npz------ | ------2.npz------ | ------3.npz------ |
                对于 [1.png, 2.png, ...] 中的每一张图片，向左向右各能获得一组event，信息是完备的
                对于0.png, 无法获取左侧的event
                '''
                ev_save_path = glob.glob(
                    os.path.join(EVENT_DIR, vid_name, "ev_npz/ev_{:06}-*".format(clip_start + _offset + 1)))
                assert ev_save_path.__len__() == 1
                ev_save_path_rel = ev_save_path[0].replace(PROCESSED_DIR, '../../../..')
                os.system(f"ln -s {ev_save_path_rel} {sample_event_dir}")

            evnpz_list = sorted(glob.glob(os.path.join(str(sample_dir), 'event/*.npz')))

            # blur=11, frame_step=8
            save_gs_only(clip, gs_start=0, blur_factor=11, sample_dir=sample_dir)
            save_gs_only(clip, gs_start=8, blur_factor=11, sample_dir=sample_dir)
            save_vg_only(gs_start=0, blur_factor=CLIP_LEN, sample_dir=sample_dir, evnpz_list=evnpz_list)

            clip_start += CLIP_STEP


def densify_clip(clip, factor=6, mode='replica'):
    """ densify the clip """
    if mode == 'replica':
        densified_clip = []
        for t in clip:
            densified_clip.extend([t] * factor)
    elif mode == 'ifrnet':
        from tools.ifrnet.ifrnet_vfi import densify_clip_ifrnet
        densified_clip = densify_clip_ifrnet(clip, factor=8)
    else:
        raise NotImplementedError
    return densified_clip


def save_gs_frame_with_vg(clip, gs_start, blur_factor, sample_dir, evnpz_list):
    """  """
    assert blur_factor in blur_candidates
    gs_span = 1  # forever 1
    num_bins = N_BINS

    ''' synthesize blurry image '''
    gs_frame = np.stack(densify_clip(clip[gs_start:gs_start+gs_span+(blur_factor-1)], factor=8, mode='ifrnet'), axis=0).mean(0).astype(np.uint8)

    # # for debug
    # s = clip[gs_start:gs_start + gs_span + (blur_factor - 1)]
    # t = densify_clip(clip[gs_start:gs_start + gs_span + (blur_factor - 1)], factor=8, mode='ifrnet')
    # ss = np.stack(s, axis=0).mean(0).astype(np.uint8)
    # tt = np.stack(t, axis=0).mean(0).astype(np.uint8)
    # for m in s:
    #     imshow(m.astype(np.uint8))
    # for m in t:
    #     imshow(m.astype(np.uint8))
    # imshow(ss, 'syn_by_sharp')
    # imshow(tt, 'syn_by_vfi')
    # for i in range(len(s)):
    #     imwrite("sharp_syn_{:03}.png".format(i), np.stack(s[:i+1], axis=0).mean(0).astype(np.uint8))
    # for i in range(len(t)):
    #     imwrite("vfi_syn_{:03}.png".format(i), np.stack(s[:i+1], axis=0).mean(0).astype(np.uint8))

    # save synthesized blurry image with corresponding voxel grid
    gs_save_path = os.path.join(sample_dir, "gs_{:06}_{:06}_{}.png".format(gs_start, gs_start, blur_factor))
    imwrite(gs_save_path, gs_frame)
    vg = get_voxel_grid(evnpz_list, num_bins, gs_start, gs_span, blur_factor)

    # compress and save
    npz_name = 'vg_{}x{}_{}to{}_{}bins.npz'.format(H, W, gs_start, gs_start+gs_span+blur_factor-1-1, num_bins)
    np.savez_compressed(os.path.join(str(sample_dir), npz_name), vg)
    print("Saving {}".format(os.path.join(str(sample_dir), npz_name)))


def save_gs_only(clip, gs_start, blur_factor, sample_dir):
    assert blur_factor in blur_candidates
    gs_span = 1  # forever 1

    ''' synthesize blurry image '''
    gs_frame = np.stack(densify_clip(clip[gs_start:gs_start+gs_span+(blur_factor-1)], factor=8, mode='ifrnet'), axis=0).mean(0).astype(np.uint8)
    gs_save_path = os.path.join(sample_dir, "gs_{:06}_{:06}_{}.png".format(gs_start, gs_start, blur_factor))
    imwrite(gs_save_path, gs_frame)


def save_vg_only(gs_start, blur_factor, sample_dir, evnpz_list):
    num_bins = N_BINS
    gs_span = 1
    vg = get_voxel_grid(evnpz_list, num_bins, gs_start, gs_span, blur_factor)
    # compress and save
    npz_name = 'vg_{}x{}_{}to{}_{}bins.npz'.format(H, W, gs_start, gs_start+gs_span+blur_factor-1-1, num_bins)
    np.savez_compressed(os.path.join(str(sample_dir), npz_name), vg)
    print("Saving {}".format(os.path.join(str(sample_dir), npz_name)))


def get_voxel_grid(evnpz_list, n_bins, rs_start, rs_span, blur_factor):

    # the corresponding rs frame is synthesized with rs_span+blur_factor-1 frames
    # for example:
    #   rs([0, 1, 2, 3], blur=3) is synthesized with gs(0), gs(1), gs(2), gs(3), gs(3+1), gs(3+2)
    #   the corresponding events are ev(1), ev(2), ev(3), ev(4), ev(5), i.e. ev[0:5]
    #   which is rs_span + (blur - 1) - 1 = 4 + 3 - 1 - 1 = 5 in total
    #   i.e. ev[0:rs_span + (blur - 1) - 1]
    evnpz_list = evnpz_list[rs_start:rs_start+rs_span+blur_factor-1-1]

    # generate vg
    _vg = generate_voxel_grid(evnpz_list, n_bins, W, H)
    return _vg


if __name__ == '__main__':
    run_gopro_process()
