## News
- 2023.7.17 - 🤣🤣🤣 **Our paper ["Neural Image Re-Exposure"](https://arxiv.org/abs/2305.13593) has been rejected by ICCV 2023** 🤪🤪🤪. 
   We temporarily release this code for a better understanding of our paper, 
   (specially for the understanding of `Neural Film`, `Neural Shutter`, and `Exposure Module`.)
   As our work remains to be improved, and we have some 
   follow-up works in the full version, the released version is trimmed and has not been tested. 
   It may lack components for deployment, which will be fixed and re-arranged in a month or two.
   By then, a new version of our code together with our revised version of paper will be released.
``

## Important modules
The core module of our **NIRE** model are implemented in `models/archs/NIRE_arch.py` and `module/temporalize_tsfm.py`.
We recommend read these codes for better understanding of our framework and method.


## Environment Setup

The code requires:
- RTX2080Ti GPU (11G Memory)
- Python 3.8
- Pytorch 1.11.0
- torchvision 0.12.0
- cudatoolkit 11.3
  

```
apt install libgl1 libglib2.0-dev  # may miss this package in docker container 
conda create -n eventinr python=3.8
conda activate eventinr
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install matplotlib opencv-python pillow tqdm pyyaml tensorboard imageio scikit-image numba einops
pip install argcomplete engineering_notation easygui numba h5py screeninfo  # for the event simulator
# pip install av  # (optional) for parsing aedat data
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
```

## Quick Start

```
python run_nire.py
```