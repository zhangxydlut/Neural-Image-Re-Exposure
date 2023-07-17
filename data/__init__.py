import math
import torch

import logging
logger = logging.getLogger('base')


def build_train_dataset(opt):
    """  """
    rank = -1 if not torch.distributed.is_initialized() else torch.distributed.get_rank()
    dataloader_list = []
    sampler_list = []
    total_epochs = 10
    total_iters = int(opt['train']['niter'])

    if not set(opt['task']) == set(k for k in opt['datasets'].keys()):
        logger.warning(f"The model is designed for tasks {sorted(set(opt['task']))}")
        logger.warning(f"There are training loaders for tasks {sorted(set(k for k in opt['datasets'].keys()))}")

    tasks = []
    for task, dataset_opt in opt['datasets'].items():
        dataset_opt['task'] = task
        tasks.append(task)
        loader_name = dataset_opt['name']
        dataset_cls, loader_builder = dataloader_registry(loader_name)

        dataloader, sampler = loader_builder(dataset_opt, dataset_cls)

        dataloader_list.append(dataloader)
        sampler_list.append(sampler)

        iter_per_epoch = int(math.ceil(len(dataloader) / dataset_opt['batch_size']))
        _total_epochs = int(math.ceil(total_iters / iter_per_epoch))
        if _total_epochs > total_epochs:
            total_epochs = _total_epochs

        if rank <= 0:
            logger.info('Number of train images: {:,d}'.format(len(dataloader)))
            logger.info('Total epochs needed: {:d} for iters {:,d}'.format(total_epochs, total_iters))

    def sampler_seed_updater(epoch):
        for s in sampler_list:
            s.update_seed(epoch)

    logger.info([d.dataset.__class__.__name__ for d in dataloader_list])

    return tasks, dataloader_list, sampler_seed_updater, total_iters, total_epochs


def dataloader_registry(name):
    if name == 'FECubeDatasetMT':  # Unrolling Dataset for Gev-RS
        from data.dataset_FECube import FECubeDataset, create_train_dataloader
        dataset_cls = FECubeDataset
        loader_builder = create_train_dataloader

    elif name == 'TimeCvtDataset':
        from data.dataset_TimeCvt import TimeCvtDataset, create_train_dataloader
        dataset_cls = TimeCvtDataset
        loader_builder = create_train_dataloader

    elif name == 'VFIGoProDataset':
        from data.dataset_VFI import VFIGoProDataset, create_train_dataloader
        dataset_cls = VFIGoProDataset
        loader_builder = create_train_dataloader

    elif name == 'UnrollGevRSDataset':
        from data.dataset_Unroll import UnrollGevRSDataset, create_train_dataloader
        dataset_cls = UnrollGevRSDataset
        loader_builder = create_train_dataloader

    elif name == 'DeblurGoProDataset':
        from data.dataset_Deblur import DeblurGoProDataset, create_train_dataloader
        dataset_cls = DeblurGoProDataset
        loader_builder = create_train_dataloader

    elif name == 'DeblurNAFNetLMDB':  # RGB-based Deblur Dataset for GoPro
        from data.dataset_NAFNet import DeblurNAFNetLMDB, create_train_dataloader
        dataset_cls = DeblurNAFNetLMDB
        loader_builder = create_train_dataloader

    elif name == 'DeblurRGBDataset':  # RGB-based Deblur Dataset for GoPro
        from data.dataset_Deblur import DeblurRGBDataset, create_train_dataloader
        dataset_cls = DeblurRGBDataset
        loader_builder = create_train_dataloader

    else:
        raise NotImplementedError("Unknown data type {}".format(name))

    return dataset_cls, loader_builder
