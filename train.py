"""
Training script of the project.

Usage:
python -m torch.distributed.launch --nproc_per_node=1 train.py \
-opt options/train/train_NIRE.yml \
--launch=pytorch \
--auto_resume \
--overwrite n_workers 2 \
--overwrite batch_size 2 \
--overwrite pretrain_model experiments/AnyPixEF5-MT2/models/45000_G.pth \
--overwrite schedule_scale 1x \
--overwrite exp_name AnyPixEF5-MT
"""
import os
import math
import argparse
import logging
import random
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import options as opt_util
import utils
from data import build_train_dataset
from models.engines import build_engine

os.chdir(os.path.abspath(os.path.dirname(__file__)))
utils.set_random_seed(123)
torch.backends.cudnn.benckmark = True
# torch.backends.cudnn.deterministic = True


def init_dist(backend='nccl', **kwargs):
    """ initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])  # TODO: check is deprecated code
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    parser.add_argument('--ext_storage', type=str, default='/media/zxy/AnyPixEF/Dataset', help='Path to external storage')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--auto_resume', action='store_true', help="Automatically resume from previous experiment")
    parser.add_argument('--runtime_val', action='store_true', help="Perform runtime validation")
    parser.add_argument('--overwrite', nargs='*', action='append', default=[], help="Overwrite items of the options")
    args = parser.parse_args()

    overwirte_items = dict()
    for k, v in (args.overwrite):
        overwirte_items[k] = v
    args.overwrite = overwirte_items
    return args


def build_opt(args, rank, is_train=True):
    opt_path = args.opt
    auto_resume = args.auto_resume
    overwrite_items = args.overwrite

    import yaml
    from utils import OrderedYaml
    Loader, Dumper = OrderedYaml()

    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)

    # =============== Options Overwrite ===============
    if 'exp_name' in overwrite_items.keys():
        opt['name'] = overwrite_items['exp_name']
    # specifying pretrained parameter
    if 'pretrain_model' in overwrite_items.keys():
        opt['path']['pretrain_model_G'] = overwrite_items['pretrain_model']
    # overwrite of batch size
    if 'batch_size' in overwrite_items.keys():
        for name, dataset_opt in opt['datasets'].items():
            dataset_opt['batch_size'] = int(overwrite_items['batch_size'])
    # overwrite crop size of training sample
    if 'crop_size' in overwrite_items.keys():
        opt['datasets']['train']['crop_size'] = int(overwrite_items['crop_size'])
    # overwrite data loader worker
    if 'n_workers' in overwrite_items.keys():
        for name, dataset_opt in opt['datasets'].items():
            dataset_opt['n_workers'] = int(overwrite_items['n_workers'])
    # schedule scaling
    schedule_scale = int(opt['train'].get('schedule_scale', '1x').strip('x'))
    if 'schedule_scale' in overwrite_items.keys():
        schedule_scale = int(overwrite_items['schedule_scale'].strip('x'))
    opt['logger']['save_checkpoint_freq'] = schedule_scale * opt['logger']['save_checkpoint_freq']
    opt['train']['niter'] *= schedule_scale
    if opt['train']['scheduler']['type'] in ['CosineAnnealingLR_Restart']:
        opt['train']['scheduler']['T_period'] = [schedule_scale * t for t in opt['train']['scheduler']['T_period']]
        opt['train']['scheduler']['restarts'] = [schedule_scale * t for t in opt['train']['scheduler']['restarts']]
    # ============================================

    # switch to training mode
    opt['is_train'] = is_train

    # path
    for key, path in opt['path'].items():
        if path and key in opt['path'] and key != 'strict_load':
            opt['path'][key] = os.path.expanduser(path)

    if opt['path'].get('root', None) is None:
        opt['path']['root'] = os.path.abspath(os.path.join(__file__, os.path.pardir))

    if is_train:
        experiments_root = os.path.join(opt['path']['root'], 'experiments', opt['name'])
        if auto_resume:
            if os.path.exists(os.path.join(experiments_root, 'models', 'latest_G.pth')) and \
                    os.path.exists(os.path.join(experiments_root, 'training_state', 'latest.state')):
                opt['path']['pretrain_model_G'] = os.path.join(experiments_root, 'models', 'latest_G.pth')
                opt['path']['resume_state'] = os.path.join(experiments_root, 'training_state', 'latest.state')
                # TODO: auto resume options from cfg.yaml
        else:
            experiments_root = utils.mkdir_safe(experiments_root, allow_create=(rank <= 0))  # create experiments_root

        # check experiment directories
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = utils.mkdir(os.path.join(experiments_root, 'models'), allow_create=(rank <= 0))
        opt['path']['training_state'] = utils.mkdir(os.path.join(experiments_root, 'training_state'), allow_create=(rank <= 0))
        opt['path']['log'] = experiments_root
        opt['path']['val_images'] = utils.mkdir(os.path.join(experiments_root, 'val_images'), allow_create=(rank <= 0))
        if opt['path']['pretrain_model_G'] is not None:
            opt['path']['pretrain_model_G'] = os.path.join(opt['path']['root'], opt['path']['pretrain_model_G']) \
                if not os.path.exists(opt['path']['pretrain_model_G']) else opt['path']['pretrain_model_G']

        # how many tasks get trained in one training iteration
        opt['train']['task_per_iter'] = opt['train'].get('task_per_iter', -1)

        # change some options for debug mode
        if 'dbg' in opt['name']:
            opt['train']['val_freq'] = 8
            opt['logger']['print_freq'] = 10
            opt['logger']['save_checkpoint_freq'] = 100

    else:  # test
        results_root = os.path.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root

    # datasets
    for task in list(opt['datasets'].keys()):
        if task not in opt['task']:
            print(f"data type {task} is not enabled in opt['task'], opt['datasets']['{task}'] will be excluded")
            opt['datasets'].pop(task)

    for task, dataset in opt['datasets'].items():
        # path to dataset
        dataset['data_root'] = [dataset['data_root']] if isinstance(dataset['data_root'], str) else dataset['data_root']
        for idx, data_root in enumerate(dataset['data_root']):
            if not os.path.exists(data_root):
                data_root = os.path.join(opt['path']['root'], data_root)
            assert os.path.exists(data_root)
            data_root = os.path.abspath(data_root)
            dataset['data_root'][idx] = data_root

    # volatile options
    opt['tb_logger'] = None  # only valid for master process (rank<=0)
    opt['dist'] = False  # default set to False, may change in main stream

    if rank <= 0:
        with open(os.path.join(opt['path']['log'], 'cfg.yml'), 'w', encoding='utf-8') as f:
            yaml.dump(opt, f, Dumper=Dumper, encoding='utf-8', allow_unicode=True)
    return opt


def main():
    # >>> options
    args = parse_args()

    # >>> distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        is_dist = False
        rank = -1
        print('Disabled distributed training.')
    else:
        is_dist = True
        init_dist()
        rank = torch.distributed.get_rank()

    opt = build_opt(args, rank, is_train=True)
    opt['dist'] = is_dist

    # >>> mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        # config loggers. Before it, the log will not work
        utils.setup_logger('base', opt['path']['log'], 'train-' + opt['name'], level=logging.INFO, screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(opt_util.dict2str(opt))  # logger.info(pprint.pformat(opt))

        # tensorboard logger
        if opt['use_tb_logger']:
            from torch.utils.tensorboard import SummaryWriter
            tb_logger = SummaryWriter(log_dir='{}/tb_logger/'.format(opt['path']['log']))
            opt['tb_logger'] = tb_logger
    else:
        utils.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # >>> finalize options
    opt = opt_util.dict_to_nonedict(opt)  # convert to NoneDict, which returns None for missing keys

    # >>> build engine
    engine = build_engine(opt)

    # >>> build dataloader
    train_loader_tasks, train_loader_list, sampler_seed_updater, total_iters, total_epochs = build_train_dataset(opt)

    # >>> loading pretrained or resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        opt_util.check_resume(opt, resume_state['iter'])  # check resume options

        # resuming training states
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(resume_state['epoch'], resume_state['iter']))
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        engine.resume_training(resume_state)  # handle optimizers and schedulers

    else:
        # when the resuming is not available, establish new training directory
        current_step = 0
        start_epoch = 0

    # >>> training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            sampler_seed_updater(epoch+rank)  # update sampler; different seed for different worker

        for iter_id, train_data_list in enumerate(zip(*train_loader_list)):
            current_step += 1
            if current_step > total_iters:
                break
            # update learning rate
            engine.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            # select the tasks for this iteration
            if opt['train']['task_per_iter'] > 0:
                loader_choices = random.choices(list(range(len(train_data_list))), k=opt['train']['task_per_iter'])
            else:
                loader_choices = list(range(len(train_data_list)))

            for c in loader_choices:
                # training
                train_data = train_data_list[c]
                iter_task = train_loader_tasks[c]  # the task for this training iteration
                engine.feed_data(train_data)  # Compare GoProEvent's dataloader with VideoINR's dataloader
                engine.optimize_parameters()

                # log
                if current_step % opt['logger']['print_freq'] == 0:
                    logs = engine.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}, task: {}, lr:('.format(epoch, current_step, iter_task)
                    for v in engine.get_current_learning_rate():
                        message += '{:.3e},'.format(v)
                    message += ')> '
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)

                    # tensorboard logger
                    if opt['use_tb_logger'] and opt['tb_logger']:
                        if rank <= 0:
                            engine.flush_tb_logger(opt['tb_logger'], current_step + c)
                    if rank <= 0:
                        logger.info(message)

            # save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:  # change to epoch-wise checkpoint
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model_path = engine.save(current_step)
                    engine.save_training_state(epoch, current_step)
                    if args.runtime_val:
                        for task in opt['task']:
                            eval_model(model_path, task)  # runtime validation

        # build two for-loops
        else:
            continue
        break

    if rank <= 0:
        logger.info('Saving the final model.')
        engine.save('final')
        logger.info('End of training.')


def eval_model(model_path, task):
    if task == 'unroll':
        # evaluate unroll performance
        torch.cuda.empty_cache()
        eval_cmd = f"python test_anypix.py " \
                   f"--mode=test " \
                   f"--test_opt=options/test/GevRS-Unroll.yml " \
                   f"--model_path={model_path}"
        os.system(eval_cmd)

    elif task == 'deblur':
        # evaluate deblur performance
        torch.cuda.empty_cache()
        eval_cmd = f"python test_anypix.py " \
                   f"--mode=test " \
                   f"--test_opt=options/test/GoPro-Deblur-720p.yml " \
                   f"--model_path={model_path}"
        os.system(eval_cmd)

    elif task == 'vfi':
        # evaluate 7-skip VFI performance
        torch.cuda.empty_cache()
        eval_cmd = f"python test_anypix.py " \
                   f"--mode=test " \
                   f"--test_opt=options/test/GoPro-VFI7.yml " \
                   f"--model_path={model_path}"
        os.system(eval_cmd)

        # evaluate 15-skip VFI performance
        torch.cuda.empty_cache()
        eval_cmd = f"python test_anypix.py " \
                   f"--mode=test " \
                   f"--test_opt=options/test/GoPro-VFI15.yml " \
                   f"--model_path={model_path}"
        os.system(eval_cmd)

    elif task == 'jdfi':
        # evaluate JDFI performance
        torch.cuda.empty_cache()
        eval_cmd = f"python test_anypix.py " \
                   f"--mode=test " \
                   f"--test_opt=options/test/GoPro-JDFI.yml " \
                   f"--model_path={model_path}"
        os.system(eval_cmd)

    else:
        print(f"Not able to perform runtime evaluation on {task}")


if __name__ == '__main__':
    main()
