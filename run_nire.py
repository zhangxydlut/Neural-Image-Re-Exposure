"""
Run test and evaluation on testing sets

Usage:
"""
import os
import cv2

import h5py
import numpy as np
import yaml
from evbase.io import H5ImageReader, H5EventsReader, TimelineSlicer
from evbase.repr.voxel_grid import evs2voxel
from evbase.utils.yaml_config import OrderedYaml
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

this_dir = os.path.dirname(os.path.abspath(__file__))
model_path=f'{this_dir}/NIRE_model/models/60000_G.pth'
if not os.path.exists(model_path):
    print("Downloading the NIRE model from \n"
          "\t https://drive.google.com/file/d/1bzSskeYmndYyO-YejqA6eMcMB__hZtEF/view?usp=share_link")
    # download from drive with gdown tool through https protocal
    os.system(f"gdown https://drive.google.com/uc?id=1bzSskeYmndYyO-YejqA6eMcMB__hZtEF "
              f"--output {this_dir}/NIRE_model.zip")
    # unzip the downloaded file and remove the zip file
    os.system(f"unzip {this_dir}/NIRE_model.zip -d {this_dir} && rm {this_dir}/NIRE_model.zip")
opt_file = f'{this_dir}/runtime.yml'


def build_opt(opt_path):
    # load training time options
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    # test time options
    opt['is_train'] = False
    opt['dist'] = False
    opt['path']['pretrain_model_G'] = model_path
    return opt


# build model
opt = build_opt(opt_file)  # build the option of model, ensure the model identical to training
from evbase.toolbox.nire.models.NIRE_engine import NIRE_Engine
nire = NIRE_Engine(opt)


def nire_data_feeder(evs_blob, rgb_blob):
    """
    Args:
        data: dict with keys ['src_img', 'src_tspec', 'vg', 'target_gs_imgs',
        'cube_span', 'tgt_tspec', 'selected_gs_times']
    """
    # get voxel grid
    t_0, evs, t_e = evs_blob
    vg, vg_tspec = evs2voxel(evs, num_bins=opt['n_bins'], mode='pytorch', device=device)

    # get rgb image
    ta_, src_img, _tb = rgb_blob
    src_img = torch.from_numpy(src_img / 255.).to(torch.float32).permute(2, 0, 1).to(device)[None]
    # get time spectrum for the source image
    if isinstance(ta_, np.int64) and isinstance(_tb, np.int64):
        ta_ = (ta_ - t_0)/(t_e - t_0)
        _tb = (_tb - t_0)/(t_e - t_0)
        tspec_a = ta_ * torch.ones(src_img.shape[-2:]).to(torch.float32).to(device)
        tspec_b = _tb * torch.ones(src_img.shape[-2:]).to(torch.float32).to(device)
    else:
        assert isinstance(ta_, np.ndarray) and isinstance(_tb, np.ndarray)
        ta_ = (ta_ - t_0)/(t_e - t_0)
        _tb = (_tb - t_0)/(t_e - t_0)
        tspec_a = torch.from_numpy(ta_).to(torch.float32).to(device)
        tspec_b = torch.from_numpy(_tb).to(torch.float32).to(device)
    im_tspec = torch.stack([tspec_a, tspec_b], dim=0).to(torch.float32).to(device)

    data = {'src_img': src_img[None], 'im_tspec': im_tspec[None][None], 'vg': vg[None], 'vg_tspec': vg_tspec[None]}
    return data


def run_nire(nire_data, t_spec: float, ragion=None):
    assert 0 <= t_spec <= 1
    if ragion is not None:
        tlbr = ragion
        for k in nire_data.keys():
            nire_data[k] = nire_data[k][..., tlbr[0]:tlbr[2], tlbr[1]:tlbr[3]]
    nire_data['tgt_tspec'] = torch.ones_like(nire_data['im_tspec']) * t_spec
    im_recon_seq = nire(nire_data)
    assert len(im_recon_seq) == 1  # this function support generating only one target image
    return im_recon_seq


if __name__ == '__main__':
    # Sanity check
    h5_file = '/media/zxy/SSD4T/NIRE-data-v2/v2e-GevRS-train-res640x360-24209_1_12.h5'
    with h5py.File(h5_file, 'r') as h5f:
        evs_reader = H5EventsReader(h5f, 'evs')
        rs_reader = H5ImageReader(h5f, 'rs_00')
        data_slicer = TimelineSlicer([rs_reader, evs_reader])

        # load a pair of ergb_blob
        rs_blob, evs_blob = data_slicer[0]
        for t_spec in np.linspace(0, 1, 11):
            nire_data = nire_data_feeder(evs_blob, rs_blob[0])
            im_recon_seq = run_nire(nire_data, t_spec)
            im_recon = (im_recon_seq * 255)[0].cpu().numpy().astype(np.uint8)[0].transpose(1, 2, 0)
            cv2.imshow('im_recon', cv2.cvtColor(im_recon, cv2.COLOR_BGR2RGB))
            cv2.waitKey(0)
