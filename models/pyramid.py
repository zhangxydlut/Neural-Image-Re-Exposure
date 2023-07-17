"""
Basic data structure for NIRE model
"""
import torch
import torch.nn.functional as F
from collections import OrderedDict


def make_pyramid(feat, size_list):
    """ sample a single feature map into a multi-scale feature pyramid """
    pyramid = []
    for size in size_list:
        h, w = size
        if len(feat.shape) == 5:  # if feat is a feature sequence
            pyramid.append(torch.stack(
                [F.interpolate(feat[:, i], (h, w), mode='bilinear', align_corners=False) for i in
                 range(feat.shape[1])], dim=1))
        elif len(feat.shape) == 4:
            pyramid.append(F.interpolate(feat, (h, w), mode='bilinear', align_corners=False))
    return pyramid


class SeqFeatPyramid(object):
    def __init__(self, feat_pyr):
        """
        Feature pyramid, default arranged in [lvl -> seq]
        Args:
            feat_pyr (list or SeqFeatPyramid):
        """
        feat_pyr = self._pyramid_to_list(feat_pyr)
        self.a = feat_pyr
        self.n_lvl = len(self.a)
        self.B, self.T, self.C, self.H, self.W = self.a[0].shape
        self.shape = (self.n_lvl, self.B, self.T, self.C, self.H, self.W)
        self.check_pyramid()

    def check_pyramid(self):
        pyr_shape_list = get_pyramid_shape(self.a)
        assert all(pyr_shape_list[i][:2] == pyr_shape_list[i+1][:2] for i in range(len(pyr_shape_list)-1))  # different level have same bs and seq_len
        if hasattr(self, 'te'):
            te_shape_list = get_pyramid_shape(self.te)
            assert all(a_shape[:2] == te_shape[:2] for a_shape, te_shape in zip(pyr_shape_list, te_shape_list))
            assert all(a_shape[3:] == te_shape[3:] for a_shape, te_shape in zip(pyr_shape_list, te_shape_list))

    @staticmethod
    def to_lvl_first(seq_first_feat):
        """
        Change the sequence-first feature list to lvl-first feature list
        Args:
            seq_first_feat: seq_first feature [seq -> lvl]
        """
        n_lvl = len(seq_first_feat[0])
        lvl_first_feat = []
        for i in range(n_lvl):
            lvl_first_feat.append([node[i] for node in seq_first_feat])
        lvl_first_feat = [torch.stack(lvl, dim=1) for lvl in lvl_first_feat]  # [lvl -> seq[B, T, C, H, W]]
        return lvl_first_feat

    @staticmethod
    def to_seq_first(lvl_first_feat):
        n_lvl = len(lvl_first_feat)
        B, T, C, H, W = lvl_first_feat[0].shape
        seq_first_feat = [[lvl_first_feat[l][:, t] for l in range(n_lvl)] for t in range(T)]
        return seq_first_feat

    def transpose(self):
        return self.to_seq_first(self.a)

    @staticmethod
    def _pyramid_to_list(feat_pyr):
        if isinstance(feat_pyr, SeqFeatPyramid):
            feat_pyr = feat_pyr.data
        else:
            assert isinstance(feat_pyr, list)
        return feat_pyr

    @property
    def te_base(self):
        # self.__te_base cannot be directly modified
        # the value of self.__te_base can only be modified by self.temporalize()
        return self.__te_base

    def temporalize(self, te):
        """ get time positional encoding of each feature
        Args:
            te(torch.Tensor): base time spectrum encoding in [B, T, C, H, W]
        """
        lvlsize_list = [l.shape[-2:] for l in self.a]
        self.__te_base = te
        self.te = make_pyramid(te, lvlsize_list)
        self.check_pyramid()
        return self

    @property
    def data(self):
        self.check_pyramid()
        return self.a

    def update(self, pyr, lvl_indices=None):
        if lvl_indices is None:
            assert pyr.shape[0] == self.shape[0]
            lvl_indices = list(range(pyr.shape[0]))
        for i, lvl in enumerate(lvl_indices):
            assert self.a[lvl].shape == pyr.a[i].shape  # the updated feature should have same shape
            assert (self.te_base == pyr.te_base).all()  # the feature can be updated only by the feature defined on the same time
            assert (self.te[lvl] == pyr.te[i]).all()  # the feature can be updated only by the feature defined on the same time
            self.a[lvl] = pyr.a[i]
        self.check_pyramid()
        return self

    def ext_lvl(self, pyr):
        new_lvl, B, T, C, H, W = pyr.shape[0]
        assert self.shape[1] == B
        assert self.te_base == pyr.te_base
        self.a.extend(pyr.a)
        self.te.extend(pyr.te)
        return self

    def __repr__(self):
        repr_str = f"SeqFeatPyramid(lvl={self.n_lvl}, bs={self.a[0].shape[0]}, seq_len={self.a[0].shape[1]}, base_shape={self.a[0].shape[-3:]})"
        return repr_str

    def __getitem__(self, index):
        """ get subset of the pyramid specified by the index
        Args:
            index (slice or list(slice)): indices

        Returns:
            sub_pyramid (SeqFeatPyramid): subset of the pyramid specified by the index

        Usage:
            # select all
            t = a[:]
            # select level
            t = a[0]; t = a[0:1]
            t = a[1]; t = a[1:2]
            # slice levels
            t = a[0:2]
            # slice the batch or seq_len
            t = a[0:2, :, 1:4]  # [lvl, bs, seq_len]
        """
        if isinstance(index, int):
            lvl_index = slice(index, index+1, None)  # keep the dim
            sub_pyr = SeqFeatPyramid(self.a[lvl_index])
            if hasattr(self, 'te_base'):
                sub_pyr.temporalize(self.te_base)
        elif isinstance(index, slice):  # select specific level(s)
            lvl_index = index
            sub_pyr = SeqFeatPyramid(self.a[lvl_index])
            if hasattr(self, 'te_base'):
                sub_pyr.temporalize(self.te_base)
        elif isinstance(index, (list, tuple)):  # select specified region of specific level(s)
            index = [idx if isinstance(idx, slice) else slice(idx, idx+1, None) for idx in index]
            lvl_idx = index[0]  # lvl
            remain_idx = index[1:3]  # [B, T]
            sub_pyramid = [l[remain_idx] for l in self.a[lvl_idx]]
            sub_pyr = SeqFeatPyramid(sub_pyramid)
            if hasattr(self, 'te_base'):
                sub_pyramid_tebase = self.te_base[remain_idx]
                sub_pyr.temporalize(sub_pyramid_tebase)
        else:
            raise NotImplementedError("index of unrecognized type")

        return sub_pyr


class CombinedPyramid(object):
    def __init__(self, **kwargs):
        self.pyr_dict = OrderedDict(**kwargs)
        self.combined_pyr = self.build_combined(self.pyr_dict)

    @property
    def shape(self):
        return self.combined_pyr.shape

    @property
    def data(self):
        return self.combined_pyr.data

    @property
    def te(self):
        return self.combined_pyr.te

    @property
    def te_base(self):
        return self.combined_pyr.te_base

    def build_combined(self, pyr_dict):
        L, B, _, C, H, W = self._check_pyr_dict(pyr_dict)
        combined_pyr = []
        for l in range(L):
            combined_pyr.append(torch.cat([pyr.data[l] for k, pyr in pyr_dict.items()], dim=1))
        combined_pyr = SeqFeatPyramid(combined_pyr)

        if any(hasattr(v, 'te') for k, v in pyr_dict.items()):
            assert all(hasattr(v, 'te') for k, v in pyr_dict.items())
            combined_te = torch.cat([pyr.te_base for k, pyr in pyr_dict.items()], dim=1)
            combined_pyr.temporalize(combined_te)

        return combined_pyr

    def _check_pyr_dict(self, pyr_dict):
        assert all(isinstance(pyr, SeqFeatPyramid) for k, pyr in pyr_dict.items())
        L, B, _T, C, H, W = pyr_dict[list(pyr_dict.keys())[0]].shape
        for k, pyr in pyr_dict.items():
            assert isinstance(pyr, SeqFeatPyramid)
            _L, _B, _, _C, _H, _W = pyr.shape
            assert _L == L, _B == B and _C == C and _H == H and _W == W
        return L, B, _T, C, H, W

    def get(self, key: str):
        print("Warning: getting feature from self.pyr_dict, in which the feature is not updated!")
        return self.pyr_dict[key]

    def update(self, pyr, lvl_indices=None):
        assert isinstance(pyr, SeqFeatPyramid)
        self.combined_pyr.update(pyr, lvl_indices)
        return self

    def ext_lvl(self, pyr):
        new_lvl, B, T, C, H, W = pyr.shape[0]
        assert self.shape[1] == B
        self.combined_pyr.ext_lvl(pyr)
        return self

    def ext_seq(self, **kwargs):
        for pyr_name, pyr in kwargs.items():
            if pyr_name in self.pyr_dict.keys():
                print(f"Warning: {pyr_name} exists in self.pyr_dict, will be updated with new value")
            self.pyr_dict[pyr_name] = pyr
        self.combined_pyr = self.build_combined(self.pyr_dict)
        return self

    def split(self):
        T = 0
        split_dict = {}
        for k, pyr in self.pyr_dict.items():
            split_dict[k] = self.combined_pyr[:, :, T:T+pyr.shape[2]]
            T += pyr.shape[2]
        return split_dict

    def transpose(self):
        return self.combined_pyr.transpose()

    def __repr__(self):
        data_info = self.combined_pyr.__repr__()
        split_info = ''.join([f"\n| {k}:\t{v} | " for k, v in self.pyr_dict.items()])
        return data_info + split_info

    def __getitem__(self, index):
        if isinstance(index, tuple) and isinstance(index[0], str):
            up2date_pyr_dict = self.split()
            return CombinedPyramid(**{k: up2date_pyr_dict[k] for k in index})
        else:
            return self.combined_pyr.__getitem__(index)


def get_pyramid_shape(pyr):
    """
    Args:
        pyr [list]: [lvl -> seq]

    Returns:
        a list of the shapes of each level in the pyramid
    """
    shape_list = []
    for lvl in pyr:
        shape_list.append(lvl.shape)
    return shape_list


if __name__ == '__main__':
    B = 8
    N = 6
    C = 32
    H = 160
    W = 240
    
    feat_pyramid = [
            torch.rand(B, N, C, H, W),
            torch.rand(B, N, C * 2, int(1/2 * H), int(1/2 * W)),
            torch.rand(B, N, C * 4, int(1/4 * H), int(1/4 * W))
        ]

    a = SeqFeatPyramid(feat_pyramid)

    # test index
    t = a[:]
    print('a[:]', t)
    t = a[0]
    print('a[0]', t)
    t = a[1]
    print('a[1]', t)
    t = a[0:1]
    print('a[0:1]', t)
    t = a[0:2, :, 1:4]  # [lvl, bs, seq_len]
    print('a[0:2, 1:3, 1:4]', t)

    # test transpose
    t = a.transpose()
    print('a.transpose()', len(t))

    # test super pyramid
    b = SeqFeatPyramid(feat_pyramid)
    c = SeqFeatPyramid(b)
    print('SeqFeatPyramid(feat_pyramid)', b)
    print('SeqFeatPyramid(b)', c)

    # test combine, update and split
    d = CombinedPyramid(b=b, c=c)
    # d.update(d.combined_pyr)
    # print('d.split()', d.split())

    # test keyword index
    print(d['b', 'c'])

    # test ext_seq
    d.ext_seq(a=a)
    print("d.ext_seq", d)
