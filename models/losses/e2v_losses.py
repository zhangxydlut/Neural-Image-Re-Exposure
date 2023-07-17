"""
losses used by SPADE-E2VID
The code is modified from
https://github.com/RodrigoGantier/SPADE_E2VID/blob/817189ad2c9e84485ec5f395f9b3305fc2d02a52/e2v_utils.py
"""
import torch
import torch.nn as nn
from scipy.stats import t
import torchvision
# from utils.spynet import run as spynet
import torch.nn.functional as F

import torch
from torch.autograd import Variable
from math import exp


class SSIM(torch.nn.Module):
    """ Modified from https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py """
    @staticmethod
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    @staticmethod
    def create_window(window_size, channel):
        _1D_window = SSIM.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    @staticmethod
    def _ssim(img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = SSIM.create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = SSIM.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return SSIM._ssim(img1, img2, window, self.window_size, channel, self.size_average)


class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class E2VIDLoss(nn.Module):
    def __init__(self):
        super(E2VIDLoss, self).__init__()
        self.vgg = VGG19().cuda().eval()
        self.pix_crit_l1 = nn.L1Loss(size_average=True)
        self.pix_crit_mse = nn.MSELoss()
        self.ssim_module = SSIM()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        # self.flownet = spynet.Network().cuda().eval()
        self.alpha = 50

    # def tem_loss(self, pred0, pred1, img0, img1):
    #     with torch.no_grad():
    #         flow = self.flownet(img1.detach(), img0.detach())  # Backward optical flow
    #         img0_warp = spynet.backwarp(img0.detach(), flow)
    #         pred0_warp = spynet.backwarp(pred0, flow)
    #         noc_mask = torch.exp(-self.alpha * torch.sum(img1.detach() - img0_warp, dim=1).pow(2)).unsqueeze(1)
    #
    #     temp_loss = self.pix_crit_l1(pred1 * noc_mask, pred0_warp * noc_mask)
    #
    #     return temp_loss

    def loss3(self, pred, y):
        # the loss function contain
        # pixel wise loss, reg loss, features loss, style loss

        # -------SSIM loss------
        ssim_loss = 1 - self.ssim_module(pred, y)
        # -------pixel wise loss-------
        pixel_loss = self.pix_crit_l1(pred, y)
        # -------features and style loss------
        y = (y * 2) - 1
        pred = (pred * 2) - 1
        features_loss, style_loss = self.featStyleLoss(pred, y.detach())

        return pixel_loss + ssim_loss + features_loss + style_loss, features_loss

    def forward(self, pred, y):
        if pred.shape[1] == 1:
            pred = torch.cat([pred]*3, dim=1)
            y = torch.cat([y]*3, dim=1)
        # the loss function contain
        # pixel wise loss, reg loss, features loss, style loss

        # -------SSIM loss------
        ssim_loss = 1 - self.ssim_module(pred, y)
        # -------pixel wise loss-------
        pixel_loss = self.pix_crit_l1(pred, y)
        # -------features and style loss------
        y = (y * 2) - 1
        pred = (pred * 2) - 1
        features_loss, style_loss = self.featStyleLoss(pred, y)

        return pixel_loss + ssim_loss + features_loss + style_loss

    def loss2(self, pred, y):
        # the loss function contain
        # pixel wise loss, reg loss, features loss, style loss
        with torch.no_grad():
            y = (y * 2) - 1
        # -------features and style loss------
        features_loss, style_loss = self.featStyleLoss(pred, y.detach())
        # -------SSIM loss------
        y = (y + 1) / 2
        pred = (pred + 1) / 2
        ssim_loss = 1 - self.ssim_module(pred, y)
        # -------pixel wise loss-------
        pixel_loss = self.pix_crit_l1(pred, y)

        return pixel_loss + ssim_loss + features_loss + style_loss, features_loss

    def gram_matrix(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (h * w)
        return gram

    def featStyleLoss(self, pred, y):
        x_vgg, y_vgg = self.vgg(pred), self.vgg(y.detach())
        f_loss = 0
        s_loss = 0
        for i in range(len(x_vgg)):
            f_loss += self.weights[i] * self.pix_crit_l1(x_vgg[i], y_vgg[i].detach())
            s_loss += self.weights[i] * self.pix_crit_l1(self.gram_matrix(x_vgg[i]),
                                                         self.gram_matrix(y_vgg[i].detach()))
        return f_loss, s_loss

    def pixelwise(self, pred, y):
        bs = pred.shape[0]
        pixel_loss = torch.pow(pred.view(bs, -1) - y.view(bs, -1), 2)
        pixel_loss = pixel_loss.mean(1)
        # pixel_loss = torch.sqrt(pixel_loss)
        pixel_loss = pixel_loss.mean()
        return pixel_loss

    def warping(self, x, flo):

        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)

        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)

        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()

        vgrid = grid + flo

        ## scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)

        output = torch.nn.functional.grid_sample(x, vgrid, align_corners=False)
        mask = torch.ones(x.size()).cuda()
        mask = torch.nn.functional.grid_sample(mask, vgrid, align_corners=False)

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask


class E2VIDLossPlus(E2VIDLoss):
    def __init__(self, ext_loss):
        super(E2VIDLossPlus, self).__init__()
        self.ext_loss = ext_loss
        self.loss_record = dict()

    def forward(self, pred, y):
        if pred.shape[1] == 1:  # compatible with gray image
            pred = torch.cat([pred]*3, dim=1)
            y = torch.cat([y]*3, dim=1)

        # -------SSIM loss------
        ssim_loss = 1 - self.ssim_module(pred, y)

        # -------pixel wise loss-------
        pixel_loss = self.pix_crit_l1(pred, y)

        # -------features and style loss------
        y = (y * 2) - 1
        pred = (pred * 2) - 1
        features_loss, style_loss = self.featStyleLoss(pred, y)

        # -------psnr loss------
        ext_loss = 0
        for ext_loss_fn in self.ext_loss:
            ext_loss += ext_loss_fn(pred, y)

        self.loss_record['l1_loss'] = pixel_loss
        self.loss_record['ssim_loss'] = ssim_loss
        self.loss_record['features_loss'] = features_loss
        self.loss_record['style_loss'] = style_loss
        self.loss_record['psnr_loss'] = ext_loss

        return pixel_loss + ssim_loss + features_loss + style_loss + ext_loss
