"""
Modified from models/VideoSR_base_model.py
"""
import math
import torch.nn.functional as F

import torch
from models.losses.nire_losses import NIRELoss
from models.engines.base_engine import BaseEngine


import logging
logger = logging.getLogger('base')


def build_loss(opt):
    logger.info(f"Using loss: {opt['loss']}")
    loss = NIRELoss()
    return loss


def define_G(opt):
    model = opt['model']

    if model == 'NIRE':
        from models.archs.NIRE_arch import AnyPixEF
        net_G = AnyPixEF(opt)
    else:
        raise NotImplementedError

    return net_G


class NIRE_Engine(BaseEngine):
    def define_G(self, opt):
        return define_G(opt)

    def build_loss(self):
        # build loss
        self.loss = build_loss(self.opt).to(self.device)

    def feed_data(self, data):
        """
        Args:
            data: dict with keys ['src_img', 'src_tspec', 'vg', 'target_gs_imgs',
            'cube_span', 'tgt_tspec', 'selected_gs_times']
        """
        self.src_img = data['src_img'].to(self.device)  # src image
        self.im_tspec = data['src_tspec'].to(self.device)  # src image
        self.vg = data['vg'].to(self.device)  # time spectrum of target image
        self.vg_tspec = data['vg_tspec'].to(self.device)  # time spectrum of target image
        self.tgt_tspec = data['tgt_tspec'].to(self.device)  # time spectrum of target image
        if self.is_train:
            self.gt = data['target_gs_imgs'].to(self.device)

    def optimize_parameters(self):
        if not self.netG.training:
            print("self.netG is NOT in training state! Converting to train ...")
            self.netG.train()
        assert self.netG.training  # make sure the model is in training state
        self.optimizer_G.zero_grad()

        # forward pass
        self.pred, aux_out = self.netG(self.src_img, self.im_tspec, self.vg, self.vg_tspec, self.tgt_tspec)

        # calc loss
        loss = 0
        for i in range(self.gt.shape[1]):
            recon_loss = self.loss(self.pred[:, i], self.gt[:, i])
            loss += recon_loss / self.gt.shape[1]

        if aux_out is not None:
            aux_loss_func = self.netG.module.calc_aux_loss if hasattr(self.netG, 'module') else self.netG.calc_aux_loss
            loss += aux_loss_func(aux_out)

        # backward pass
        loss.backward()

        ## find unused parameters
        # from utils.misc import find_unused_params
        # find_unused_params(self.netG)  # exclude unused params for parallel training

        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), 0.01)

        self.optimizer_G.step()

        # set log
        self.log_dict['loss'] = loss.item()
        if hasattr(self.loss, 'loss_record'):
            for k, v in self.loss.loss_record.items():
                self.log_dict[k] = v.item()

    def test(self, data, mode='full'):
        if mode == 'full':
            return self.infer(data)
        elif mode == 'grid':
            return self.infer_grid(data)
        else:
            raise NotImplementedError

    def infer(self, data):
        if self.netG.training:
            print("self.netG is in TRAINING state! Converting to eval ...")
            self.netG.eval()
        assert not self.netG.training  # make sure the model is in eval state
        with torch.no_grad():
            # load and pad RGB image and concomitant time spectrum
            src_img = data['src_img']
            im_tspec = data['src_tspec']
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
                _pred, _ = self.netG(src_img, im_tspec, vg, vg_tspec, _tgt_tspec)
                # _pred, _ = self.netG(src_img[:, :, :, :, 160:480], im_tspec[:, :, :, :, 160:480], vg[:, :, :, 160:480], vg_tspec[:, :, :, 160:480], _tgt_tspec[:, :, :, :, 160:480])
                pred.append(_pred.cpu())
                # torch.cuda.empty_cache()
            pred = torch.cat(pred, dim=1)
            pred = pred[:, :, :, :H, :W]
        return pred
