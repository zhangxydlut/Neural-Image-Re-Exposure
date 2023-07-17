import torch
import torch.nn as nn
from models.modules.im_encoder.nafnet.naf_modules import NAFEncoder
from models.modules.e2vid_modules import build_e2vid_encoder, build_e2vid_decoder
from models.modules.temporalize_tsfmr import NerfPositionalEncoding, TemporalizeModuleMS
from models.pyramid import SeqFeatPyramid, CombinedPyramid, get_pyramid_shape

DEBUG = False
debug_cnt = 0


class SeqRevFuseMS(nn.Module):
    """ Fuse the multi-level forward/reverse hidden states of BiLSTM """
    def __init__(self, dims):
        super(SeqRevFuseMS, self).__init__()
        self.encoder_output_sizes = dims
        self.fuse_layers = nn.ModuleList()
        for enc_out_dim in self.encoder_output_sizes:
            fuse_layer = nn.Sequential(
                nn.Conv2d(in_channels=2 * enc_out_dim, out_channels=enc_out_dim, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
            )
            self.fuse_layers.append(fuse_layer)

    def forward(self, x_ms_seq, x_ms_rev):

        node_feat_list = []
        for t in range(len(x_ms_seq)):
            # transverse all seq nodes
            node_feat = []
            for lvl_fuse, x_seq, x_rev in zip(self.fuse_layers, x_ms_seq[t], x_ms_rev[t]):
                # fuse for each level
                # x_seq or x_rev: List([B, 32, H, W], [B, 64, H/2, W/2], [B, 128, H/4, W/4], [B, 256, H/8, W/8])
                lvl_bi_feat = torch.cat([x_seq, x_rev], dim=1)
                lvl_bi_feat = lvl_fuse(lvl_bi_feat)  # bi-direction fused feature of this level
                node_feat.append(lvl_bi_feat)
            node_feat_list.append(node_feat)
        return node_feat_list


def run_att_with_pyramid(att_module, q_pyr, kv_pyr, empty_cache=False):
    att_feat = att_module(q_inp=q_pyr.data, q_pe=q_pyr.te, kv_inp=kv_pyr.data, k_pe=kv_pyr.te)  # nake feature
    att_feat = SeqFeatPyramid(att_feat)
    att_feat.temporalize(q_pyr.te_base)
    if empty_cache:
        torch.cuda.empty_cache()
    return att_feat


class NeuralFilmPyramid(nn.Module):
    def __init__(self, dims):
        super(NeuralFilmPyramid, self).__init__()
        self.encoder_output_sizes = dims
        self.pix_embed = nn.ModuleList()
        for enc_out_dim in self.encoder_output_sizes:
             self.pix_embed.append(nn.Embedding(1, enc_out_dim))  # the placeholder token represent the target pixel

    def forward(self, seq_shape_list, seq_len=1):
        assert len(self.pix_embed) == len(seq_shape_list)
        vir_feat_pyr = []
        for pix_embed, seq_shape in zip(self.pix_embed, seq_shape_list):
            B, _, C, H, W = seq_shape
            assert C == pix_embed.weight.shape[1]
            vir_feat_pyr.append(pix_embed.weight[None, :, :, None, None].repeat(B, 1, 1, H, W).squeeze(1))
        return [vir_feat_pyr] * seq_len


class AnyPixEF(nn.Module):
    """ NIRE Architecture
    Architecture for Neural Image Re-Exposure
    """
    def __init__(self, opt):
        super(AnyPixEF, self).__init__()
        self.opt = opt
        self.empty_cache = not opt['is_train'] and not opt['fast_test']
        self.setup_options()
        self.build_network()

    def setup_options(self):
        self.att_window = self.opt['network'].get('att_window', (2, 2))
        self.base_channels = self.opt['network'].get('base_channel', 32)  # TODO: write to config.yml
        self.vg_dim = self.opt['network']['vg_dim']
        self.att_lvl_list = self.opt['network']['att_lvl']
        self.pe_dim = self.opt['network']['pe_dim']

    def build_network(self):
        # build layers
        self.rgb_encoder = NAFEncoder(
            img_channel=3,
            width=self.base_channels,
            enc_blk_nums=[1, 1, 1],
            middle_blk_num=2,
            dec_blk_nums=[1, 1, 1],
            mode='u-enc'
        )

        self.ev_encoder = build_e2vid_encoder(in_channels=self.vg_dim, base_channels=self.base_channels)
        self.seqrev_fuse = SeqRevFuseMS(self.ev_encoder.encoder_output_sizes)
        self.e2v_decoder = build_e2vid_decoder(self.vg_dim, pred_dim=3, base_channels=self.base_channels)
        self.vir_feat_base = NeuralFilmPyramid(self.ev_encoder.encoder_output_sizes)
        self.n_lvl = len(self.ev_encoder.encoder_output_sizes)

        # positional encoding
        self.time_encoder = NerfPositionalEncoding(depth=self.pe_dim, sine_type='lin_sine')  # TODO: is lin_sine reasonable?

        # Attention Temporalize
        lvl_dim_list = [self.ev_encoder.encoder_output_sizes[t] for t in self.att_lvl_list]
        self.virtual_init = TemporalizeModuleMS(lvl_dim_list, dim_pe=self.pe_dim * 2 * 2, att_window=self.att_window, pix_init=False, enable_dfm=True)
        self.att_layer_2 = TemporalizeModuleMS(lvl_dim_list, dim_pe=self.pe_dim * 2 * 2, att_window=self.att_window, pix_init=False, enable_dfm=True)
        self.att_layer_3 = TemporalizeModuleMS(lvl_dim_list, dim_pe=self.pe_dim * 2 * 2, att_window=self.att_window, pix_init=False, enable_dfm=True)

    def get_rel_tspec(self, tspec, ref_tspec):
        """
        We use relative time spectrum to generate time encoding.
        For example, the relative time spectrum of a real feature is
            realfeat_reltspec = realfeat_tspec - tgt_time.mean([-1, -2], keepdim=True).
            feat_te = self.time_encoder(realfeat_reltspec, axis=2)
        It worth noting that we use the massive center of tgt_tspec, i.e.  tgt_time.mean([-1, -2], keepdim=True)
        as the reference time.
        That is bcause the tgt_spec is designed to be compatible with RS output, which means the not all values
        in tgt_tspec are the same. But the reference time hs to be a time point, so we calculate the massive center.

        Args:
            tspec: normalized([0, 1]) time spectrum
            ref_tspec: normalized([0, 1]) time spectrum as reference

        Returns:
            rel_tspec
        """
        # get time encodings
        rel_tspec = self.time_encoder((tspec - ref_tspec.mean([-1, -2], keepdim=True)), axis=2)
        return rel_tspec

    def gen_ev_feat(self, vg, vg_tspec):
        vg_fwd = torch.stack(vg.split(self.vg_dim, dim=1), dim=1)
        B, n_vg, vg_bin, H, W = vg_fwd.shape

        # Bi-LSTM: forward
        ev_feat_seq = []
        prev_states = None
        for i in range(vg_fwd.shape[1]):
            block_feat, prev_states, ev_feat = self.ev_encoder(vg_fwd[:, i], prev_states)
            ev_feat_seq.append(block_feat)

        # Bi-LSTM: reversed
        vg_rev = vg.flip(1)  # (vg.flip(1)[:, 0]  == vg[:, -1]).all()
        vg_rev = torch.stack(vg_rev.split(self.vg_dim, dim=1), dim=1)
        ev_feat_rev = []
        prev_states = None
        for i in range(vg_rev.shape[1]):
            block_feat, prev_states, ev_feat = self.ev_encoder(vg_rev[:, i], prev_states)
            ev_feat_rev.append(block_feat)

        # Fuse the forward and reversed
        ev_feat_bi = self.seqrev_fuse(ev_feat_seq, ev_feat_rev)

        # ev_feat = ev_feat_seq  # for offical model debugging
        # ev_feat = ev_feat_rev  # for offical model debugging
        ev_feat = ev_feat_bi

        # get time encoding for the voxel grid
        '''
        vg_tspec > |   |   |
        vg       > :---:---:
        '''
        vg_tspec = torch.stack([vg_tspec[:, (i*self.vg_dim, (i+1)*self.vg_dim)] for i in range(n_vg)], dim=1)

        return ev_feat, vg_tspec

    def gen_im_feat(self, x):
        B, T, C, H, W = x.size()  # T input video frames
        im_feat_seq = []
        for t in range(T):
            im_feat = self.rgb_encoder(x[:, t])
            im_feat_seq.append(im_feat)
        return im_feat_seq

    def decode(self, im_feat, ev_feat, vir_feat, aux):
        """
        Args:
            im_feat:
            ev_feat:
            vir_feat:
            tgt_tspec: specified time spectrum of target image
        Returns:
            predicted image defined by tgt_tspec
        """
        n_lvl, B, _, _, H, W = vir_feat.shape

        # summary the features to init vir_feat
        real_feat = CombinedPyramid(im_feat=im_feat, ev_feat=ev_feat)
        _vir_feat_ = run_att_with_pyramid(self.virtual_init, q_pyr=vir_feat, kv_pyr=real_feat, empty_cache=self.empty_cache)
        vir_feat.update(_vir_feat_)

        # intra-mem interaction
        all_feat = CombinedPyramid(im_feat=im_feat, ev_feat=ev_feat, vir_feat=vir_feat)
        _all_feat_ = run_att_with_pyramid(self.att_layer_2, q_pyr=all_feat, kv_pyr=all_feat, empty_cache=self.empty_cache)
        all_feat.update(_all_feat_)
        vir_feat = all_feat.split()['vir_feat']
        assert (vir_feat.data[0] == all_feat.data[0][:, -1:]).all()  # TODO: for debug

        # final summary
        _vir_feat_ = run_att_with_pyramid(self.att_layer_3, q_pyr=vir_feat, kv_pyr=all_feat, empty_cache=self.empty_cache)

        # decode the virtual frame
        _vir_feat_ = [t.flatten(0, 1) for t in _vir_feat_.data]
        pred = self.e2v_decoder(_vir_feat_).unflatten(0, (B, -1))  # [B, T, C, H, W]
        return pred, aux

    def forward(self, im, im_tspec, vg, vg_tspec, tgt_tspec, aux=None):
        # get feature from source data
        im_feat, im_tspec = self.gen_im_feat(im), im_tspec  # im_feat: [seq -> lvl]
        ev_feat, vg_tspec = self.gen_ev_feat(vg, vg_tspec)  # ev_feat: [seq -> lvl]
        ev_feat = SeqFeatPyramid(SeqFeatPyramid.to_lvl_first(ev_feat))
        im_feat = SeqFeatPyramid(SeqFeatPyramid.to_lvl_first(im_feat))
        # generate void feature for virtual frame
        vir_feat, tgt_tspec = self.vir_feat_base(get_pyramid_shape(im_feat.data)), tgt_tspec
        vir_feat = SeqFeatPyramid(SeqFeatPyramid.to_lvl_first(vir_feat))

        # decode for each target time
        preds = []
        for t in range(tgt_tspec.shape[1]):
            # get feature with pyramid
            _tgt_tspec = tgt_tspec[:, t:t + 1]
            _im_tspec = self.get_rel_tspec(im_tspec, _tgt_tspec)
            im_feat.temporalize(_im_tspec)
            _vg_tspec = self.get_rel_tspec(vg_tspec, _tgt_tspec)
            ev_feat.temporalize(_vg_tspec)
            _tgt_tspec = self.get_rel_tspec(_tgt_tspec, _tgt_tspec)
            vir_feat.temporalize(_tgt_tspec)

            pred, aux = self.decode(im_feat, ev_feat, vir_feat, aux)
            preds.append(pred)
        preds = torch.cat(preds, dim=1)

        return preds, aux

    def calc_aux_loss(self, aux):
        """ calculating auxiliary loss"""
        return 0
