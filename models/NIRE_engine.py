import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from collections import OrderedDict

# NIRE model definition
from .archs.NIRE_arch import NIRE

import logging
logger = logging.getLogger('base')


class NIRE_Engine(object):
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.cuda.current_device()
        self.is_train = opt['is_train']

        # init logging
        self.tb_logger = opt.get('tb_logger', None)
        self.log_dict = OrderedDict()

        # define network and load pretrained models
        self.net = NIRE(opt).to(self.device)

        self.load()

    def load(self):
        load_path = self.opt['path']['pretrain_model_G']
        if isinstance(self.net, nn.DataParallel) or isinstance(self.net, DistributedDataParallel):
            self.net = self.net.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        _incompatible_keys = self.net.load_state_dict(load_net_clean, strict=True)
        logger.info(_incompatible_keys)

    def __call__(self, data):
        if self.net.training:
            print("self.netG is in TRAINING state! Converting to eval ...")
            self.net.eval()
        assert not self.net.training  # make sure the model is in eval state
        with torch.no_grad():
            # load and pad RGB image and concomitant time spectrum
            src_img = data['src_img']
            im_tspec = data['im_tspec']
            B, T, C, H, W = src_img.shape
            pad = math.ceil(W/16)*16 - W, math.ceil(H/16)*16 - H
            src_img = F.pad(src_img, (0, pad[0], 0, pad[1]))
            im_tspec = F.pad(im_tspec, (0, pad[0], 0, pad[1]))  # F.interpolate(im_tspec[:, :, 0], (384, 640))[:, :, None]

            # load and pad voxel grid
            vg = data['vg']  # time spectrum of target image
            vg = F.pad(vg, (0, pad[0], 0, pad[1]))  # TODO: use F.pad
            vg_tspec = data['vg_tspec']
            vg_tspec = F.pad(vg_tspec, (0, pad[0], 0, pad[1]))  # TODO: use F.pad

            # load and pad target time spectrum
            tgt_tspec = data['tgt_tspec']  # time spectrum of target image
            tgt_tspec = F.pad(tgt_tspec, (0, pad[0], 0, pad[1]))

            pred = []
            for _tgt_tspec in tgt_tspec.chunk(tgt_tspec.shape[1], dim=1):
                _pred, _ = self.net(src_img, im_tspec, vg, vg_tspec, _tgt_tspec)
                # _pred, _ = self.net(src_img[:, :, :, :, 160:480], im_tspec[:, :, :, :, 160:480], vg[:, :, :, 160:480], vg_tspec[:, :, :, 160:480], _tgt_tspec[:, :, :, :, 160:480])
                pred.append(_pred.cpu())
                # torch.cuda.empty_cache()
            pred = torch.cat(pred, dim=1)
            pred = pred[:, :, :, :H, :W]
        return pred
