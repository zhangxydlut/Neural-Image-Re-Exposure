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
1. Download the pre-trained model from [google drive](https://drive.google.com/file/d/1bzSskeYmndYyO-YejqA6eMcMB__hZtEF/view?usp=share_link).

2. Process data with scripts in `datasets/`. Please refer to the guidances of specific scrips 
   (e.g. [script for processing gopro for deblurring, vfi, jdfi](datasets/GoPro1280x720/GoPro1280x720.md)).

3. Run the testing scrips
```
export MODEL_PATH=path/to/the/donwloaded/models/60000_G.pth

# Pick a script for testing 
python test.py --vis --fast_test --test_opt=options/test/GoPro-VFI7.yml --model_path=$MODEL_PATH
python test.py --vis --fast_test --test_opt=options/test/GoPro-VFI15.yml --model_path=$MODEL_PATH
python test.py --vis --fast_test --test_opt=options/test/GoPro-JDFI.yml --model_path=$MODEL_PATH
python test.py --vis --fast_test --test_opt=options/test/GoPro-Deblur-720p.yml --model_path=$MODEL_PATH
python test.py --vis --fast_test --test_opt=options/test/GevRS-Unroll.yml --model_path=$MODEL_PATH
```


## Training
1. Process data with scripts in `datasets/`. Please refer to the guidances of specific scrips 
   (e.g. [script for processing gopro for deblurring, vfi, jdfi](datasets/GoPro1280x720/GoPro1280x720.md)).

2. Run the training script
```
python -m torch.distributed.launch --nproc_per_node=1 train.py \
-opt options/train/nire/train_NIRE.yml \
--launch=pytorch \
--auto_resume \
--overwrite n_workers 2 \
--overwrite batch_size 2 \
--overwrite schedule_scale 1x \
--overwrite exp_name NIRE-MT
```
