""" Transformer-based temporalization module """
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NerfPositionalEncoding(nn.Module):
    def __init__(self, depth=32, sine_type='lin_sine'):
        """ out_dim = in_dim * depth * 2 """
        super().__init__()
        if sine_type == 'lin_sine':
            self.bases = [i+1 for i in range(depth)]
        elif sine_type == 'exp_sine':
            self.bases = [2**i for i in range(depth)]
        print(f'using {sine_type} as positional encoding')

    @torch.no_grad()
    def forward(self, inputs, axis):
        """
        Args:
            inputs: Normalized coordinate in [0, 1]

        Returns:
            out: NeRF positional encoding
        """
        out = torch.cat([torch.sin(i * math.pi * inputs) for i in self.bases] + [torch.cos(i * math.pi * inputs) for i in self.bases], axis=axis)
        assert torch.isnan(out).any() == False
        return out


from models.modules.ffn_layer import FeedForward
from einops import rearrange


def sample_feat(feat, flow, interp_mode='bilinear', padding_mode='zeros'):
    """ Warp an image or feature map with predicted offset (optical flow)

    Args:
        feat (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, 2, H, W), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    B, C, h, w = feat.size()
    assert feat.size()[-2:] == flow.size()[-2:]
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(feat)

    # get warped grid
    vgrid = grid + flow.permute(0, 2, 3, 1)
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0  # scale grid to [-1,1]
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0  # scale grid to [-1,1]
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    out = F.grid_sample(feat.float(), vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return out


class DfmSW_MSA(nn.Module):
    """
    Deformable Sub-window Multi-head Attention Module
    Reference: https://github.com/linjing7/VR-Baseline/blob/main/mmedit/models/backbones/sr_backbones/FGST_util.py
    """
    def __init__(
            self,
            dim,
            window_size=(2, 2),
            dim_head=64,
            heads=8,
            shift=False
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.shift = shift
        inner_dim = dim_head * heads

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Conv2d(dim, inner_dim, 3, 1, 1, bias=False)
        self.to_k = nn.Conv2d(dim, inner_dim, 3, 1, 1, bias=False)
        self.to_v = nn.Conv2d(dim, inner_dim, 3, 1, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 3, 1, 1, bias=False)

    @staticmethod
    def sample_attention_set(feat, flow, h, w, mode='bilinear'):
        feat = sample_feat(feat, flow, interp_mode=mode)
        return feat

    def forward(self, q_inp, q_pe, kv_inp, k_pe, k_flow=None):
        """
        Args:
            q_inp:  [n, 1, c, h, w]
            q_pe:  [n, 1, 1, h, w]
            kv_inp:  [n, t, c, h, w]  (t: length of ev_feat sequence)
            k_pe:  [n, t, 1, h, w]
            k_flow:   [n, t, 2, h, w], offset of key tokes

        Returns:
            out: [n,1,c,h,w]
        """
        b, f_q, c, h, w = q_inp.shape
        if k_flow is None:
            k_flow = kv_inp.new_zeros(b, kv_inp.shape[1], 2, h, w)  # TODO: how to assign offset to different tokens within the window
        assert kv_inp.shape[1] == k_flow.shape[1]  # each seq have a sampling flow

        hb, wb = self.window_size

        # sliding window
        if self.shift:
            q_inp, kv_inp = map(lambda x: torch.roll(x, shifts=(-hb // 2, -wb // 2), dims=(-2, -1)), (q_inp, kv_inp))
            if k_flow is not None:
                k_flow = torch.roll(k_flow, shifts=(-hb // 2, -wb // 2), dims=(-2, -1))

        # retrive key elements
        k_inp = []
        for i in range(kv_inp.shape[1]):
            _k, _flow = kv_inp[:, i], k_flow[:, i]
            _k = self.sample_attention_set(_k, _flow, h, w, mode='bilinear')
            k_inp.append(_k)
        k_inp = torch.stack(k_inp, dim=1)

        # norm
        q = self.norm_q(q_inp.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        kv = self.norm_kv(k_inp.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        q = self.to_q((q + q_pe).flatten(0, 1))
        k = self.to_k((kv + k_pe).flatten(0, 1))
        v = self.to_v(kv.flatten(0, 1))

        # split into (B, N, C)
        q, k, v = map(lambda t: rearrange(t, '(b f) c (h p1) (w p2)-> (b h w) (f p1 p2) c', p1=hb, p2=wb, b=b), (q, k, v))
        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        # scale
        q *= self.scale
        # attention
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        # aggregate
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        # merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        # merge windows back to original feature map
        out = rearrange(out, '(b h w) (f p1 p2) c -> (b f) c (h p1) (w p2)', b=b, h=(h // hb), w=(w // wb), p1=hb, p2=wb)
        # combine heads
        out = self.to_out(out).view(b, f_q, c, h, w)

        # inverse shift
        if self.shift:
            out = torch.roll(out, shifts=(hb // 2, wb // 2), dims=(-2, -1))

        return out


class TemporalizeLayer(nn.Module):
    def __init__(self, dim, dim_head=32, n_heads=4, dim_pe=64, att_window=(2, 2), pix_init=False, enable_dfm=False):
        super(TemporalizeLayer, self).__init__()

        # attention block
        self.att = DfmSW_MSA(dim, window_size=att_window, dim_head=dim_head, heads=n_heads)

        # project positional encoding to feature space
        self.pe_proj = nn.Conv2d(in_channels=dim_pe, out_channels=dim, kernel_size=1, stride=1, padding=0)
        # TODO: may be use kernel=3 in will be self.pe_proj better, because this make the pe slope-aware

        # flags
        self.pix_init = pix_init
        self.enable_dfm = enable_dfm

        if pix_init:
            # init virtual pixel feature
            self.pix_embed = nn.Embedding(1, dim)  # the placeholder token represent the target pixel
        if enable_dfm:
            # the layer generate offset from features
            self.offset = nn.Conv2d(in_channels=dim, out_channels=2,  # TODO: 2*att_window[0]*att_window[1]
                                    kernel_size=1, stride=1, padding=0)
        self.ffn = FeedForward(dim, num_resblocks=2)

    def forward(self, q_inp, q_pe, kv_inp, k_pe):
        """
        Args:
            q_inp:  [n, 1, c, h, w]
            q_pe:  [n, 1, 1, h, w]
            kv_inp:  [n, t, c, h, w]  (t: length of ev_feat sequence)
            k_pe:  [n, t, 1, h, w]  (t: length of ev_feat sequence)

        Returns:
            out: same dim as q_inp
        """
        B, _, C, H, W = kv_inp.shape
        q_pe = self.pe_proj(q_pe.flatten(0, 1)).unflatten(0, (B, -1))
        k_pe = self.pe_proj(k_pe.flatten(0, 1)).unflatten(0, (B, -1))
        if self.pix_init:
            assert q_inp is None
            virtual_frame = self.pix_embed.weight[None, :, :, None, None].repeat(B, q_pe.shape[1], 1, H, W)  # init the virtual frame with the learning-based pixel embedding
            q_inp = virtual_frame

        # attention
        k_flow = None
        if self.enable_dfm:
            k_flow = self.offset(kv_inp.flatten(0, 1)).unflatten(0, (B, -1))
            assert k_flow.shape[2] == 2  # only support one offset for now
            # TODO: different offset for different head/token
        out = self.att(q_inp, q_pe, kv_inp, k_pe, k_flow) + q_inp
        # FFN
        out = torch.stack(list(map(lambda x: self.ffn(x), [out[:, t] for t in range(out.shape[1])])), dim=1)
        return out


class TemporalizeModuleMS(nn.Module):
    def __init__(self, dims, dim_pe=64, att_window=(2, 2), pix_init=False, enable_up=False, enable_dfm=False):
        super(TemporalizeModuleMS, self).__init__()
        self.encoder_output_sizes = dims
        self.fuse_layers = nn.ModuleList()  # layers that perform intra-level node-wise interaction
        self.enable_up = enable_up
        self.dim_head = 32
        for enc_out_dim in self.encoder_output_sizes:
            fuse_layer = TemporalizeLayer(
                dim=enc_out_dim, dim_head=self.dim_head, n_heads=1 + enc_out_dim//self.dim_head,
                dim_pe=dim_pe,
                att_window=att_window, pix_init=pix_init, enable_dfm=enable_dfm
            )
            self.fuse_layers.append(fuse_layer)

        if self.enable_up:
            self.up_layers = nn.ModuleList()
            for enc_out_dim in list(reversed(self.encoder_output_sizes))[:-1]:
                up_layer = nn.Sequential(
                    nn.Conv2d(enc_out_dim, enc_out_dim//2, kernel_size=3, stride=1, padding=1, groups=enc_out_dim//self.dim_head),
                    nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False))
                self.up_layers.append(up_layer)

    def forward(self, q_inp, q_pe, kv_inp, k_pe):
        B, _, _, H, W = kv_inp[0].shape
        n_lvl = len(kv_inp)
        assert len(q_inp) == n_lvl and len(self.fuse_layers) == n_lvl

        lvl_feat_list = []
        for lvl_idx, (_q_inp, _q_pe, _kv_inp, _k_pe, fuse_layer) in enumerate(zip(reversed(q_inp), reversed(q_pe), reversed(kv_inp), reversed(k_pe), reversed(self.fuse_layers))):  # for each level
            if self.enable_up and lvl_idx > 0:
                # up-sample the previous level's virtual frame and fuse it with the virtual frame of current level
                _q_inp = _q_inp + self.up_layers[lvl_idx-1](lvl_feat.flatten(0, 1)).unflatten(0, (B, -1))
            lvl_feat = fuse_layer(_q_inp, _q_pe, _kv_inp, _k_pe)
            lvl_feat_list.append(lvl_feat)
        lvl_feat_list = list(reversed(lvl_feat_list))
        # lvl_feat_list = [lvl_feat_list]  # (lvl) -> (1, lvl)
        return lvl_feat_list


if __name__ == '__main__':
    T = 7
    fake_q = torch.rand(5, 1, 32, 128, 128).cuda()
    fake_k = torch.rand(5, T, 32, 128, 128).cuda()
    fake_flow = torch.rand(5, T, 2, 128, 128).cuda()

    net = DfmSW_MSA(dim=32, window_size=(T, 4, 4)).cuda()
    pred = net(fake_q, fake_k, fake_flow)
    print()
