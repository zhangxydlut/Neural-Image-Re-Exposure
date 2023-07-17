import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.lr_scheduler as lr_scheduler

import logging
logger = logging.getLogger('base')


class BaseEngine(object):
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.cuda.current_device()
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

        # check distributed training world
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        # init logging
        self.tb_logger = opt.get('tb_logger', None)
        self.log_dict = OrderedDict()

        # define network and load pretrained models
        self.netG = self.define_G(opt).to(self.device)

        # init parallel model
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)

        if self.is_train:
            self.build_loss()
        self.print_network()  # print network
        self.load()
        self.setup_engine(opt)

    def get_current_log(self):
        return self.log_dict

    def print_network(self):

        def get_network_description(network):
            """Get the string and total parameters of the network"""
            if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
                network = network.module
            s = str(network)
            n = sum(map(lambda x: x.numel(), network.parameters()))
            return s, n

        s, n = get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def freeze_model(self):
        for optimizer in self.optimizers:
            optimizer.param_groups[0]['lr'] = 0

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None and os.path.exists(load_path_G):
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
        else:
            if self.is_train:
                logger.info("No Pretrained Model Found! Start Training from Scratch...")
            else:
                raise RuntimeError("No Pretrained Model Found! Cannot perform testing")

    def save(self, iter_label):
        return self.save_network(self.netG, 'G', iter_label)

    def save_network(self, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)
        # create link
        if os.path.exists(os.path.join(self.opt['path']['models'], 'latest_G.pth')):
            os.system(f"rm {os.path.join(self.opt['path']['models'], 'latest_G.pth')}")
        os.system(f"ln -s {os.path.basename(save_path)} {os.path.join(self.opt['path']['models'], 'latest_G.pth')}")
        return save_path

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        _incompatible_keys = network.load_state_dict(load_net_clean, strict=strict)
        logger.info(_incompatible_keys)

    def save_training_state(self, epoch, iter_step):
        """Saves training state during training, which will be used for resuming"""
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        torch.save(state, save_path)
        # create link
        if os.path.exists(os.path.join(self.opt['path']['training_state'], 'latest.state')):
            os.system(f"rm {os.path.join(self.opt['path']['training_state'], 'latest.state')}")
        os.system(f"ln -s {save_path} {os.path.join(self.opt['path']['training_state'], 'latest.state')}")

    def resume_training(self, resume_state):
        """Resume the optimizers and schedulers for training"""
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def setup_engine(self, opt):
        """ establish optimizer and optimizer """
        if self.is_train:
            # train mode
            train_opt = opt['train']
            self.netG.train()

            # build optimizers
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            optim_type = train_opt['optim_g'].pop('type')
            if optim_type == 'Adam':
                self.optimizer_G = torch.optim.Adam(optim_params, **train_opt['optim_g'])
            elif optim_type == 'SGD':
                self.optimizer_G = torch.optim.SGD(optim_params, **train_opt['optim_g'])
            elif optim_type == 'AdamW':
                self.optimizer_G = torch.optim.AdamW([{'params': optim_params}], **train_opt['optim_g'])
            else:
                raise NotImplementedError(f'optimizer {optim_type} is not supperted yet.')

            self.optimizers.append(self.optimizer_G)

            # build schedulers
            scheduler_type = train_opt['scheduler'].pop('type')

            if scheduler_type == 'MultiStepLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, **train_opt['scheduler']))
            elif scheduler_type == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(optimizer, **train_opt['scheduler']))
            else:
                raise NotImplementedError()

    def update_learning_rate(self, cur_iter, warmup_iter=-1):

        def _set_lr(lr_groups_l):
            ''' set learning rate for warmup,
            lr_groups_l: list for lr_groups. each for a optimizer'''
            for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
                for param_group, lr in zip(optimizer.param_groups, lr_groups):
                    param_group['lr'] = lr

        def _get_init_lr():
            # get the initial lr, which is set by the scheduler
            init_lr_groups_l = []
            for optimizer in self.optimizers:
                init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
            return init_lr_groups_l

        for scheduler in self.schedulers:
            scheduler.step()
        # set up warm up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = _get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            _set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        lr_l = []
        for param_group in self.optimizers[0].param_groups:
            lr_l.append(param_group['lr'])
        return lr_l

    def get_current_visuals(self):
        raise NotImplementedError

    def get_current_losses(self):
        raise NotImplementedError

    def define_G(self, opt):
        raise NotImplementedError

    def build_loss(self):
        raise NotImplementedError

    def feed_data(self, data):
        raise NotImplementedError

    def optimize_parameters(self):
        raise NotImplementedError
