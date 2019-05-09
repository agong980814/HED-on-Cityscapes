import os
import sys
from time import time
import math
import argparse
import itertools
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist 
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets 
from tensorboardX import SummaryWriter 
from torchvision.utils import make_grid
from utils import AverageMeter
from loss import CombinedLoss
import models
from models.networks import init_net
from models.networks import get_norm_layer
from models.networks import GANLoss
from models.hed import *
import cv2
import random
from data import get_dataset
from models.net_utils import *
# from cfg import cfg

color_map = [
[128, 64, 128] ,   # road
[244, 35, 232]  ,  # sidewald
[70, 70, 70] ,  # building
[102, 102, 156] , # wall
[190, 153, 153] , # fence
[153, 153, 153] , # pole
[250, 170, 30] , # traffic light
[220, 220, 0] ,  # traffic sign
[107, 142, 35] , # vegetation
[152, 251, 152] , # terrain
[70, 130, 180]  , # sky
[220, 20, 60] , # person
[255, 0, 0]  , # rider
[0, 0, 142]   , # car
[0, 0, 70]  ,  # truck
[0, 60, 100] ,  # bus
[0, 80, 100] ,  # on rails / train
[0, 0, 230]  , # motorcycle
[119, 11, 32] , # bicycle
[0, 0, 0]   # None
]

def get_hned_model(args):
    modelPath = '/home/agong/HED-on-Cityscapes/model/network-bsds500.pytorch'
    hed = models.__dict__['HNED']().cuda(args.rank)
    hed.load_state_dict(torch.load(modelPath))

    lr = 1e-6
    
    fuse_params = list(map(id, hed.moduleCombine.parameters()))
    conv5_params = list(map(id, hed.moduleVggFiv.parameters()))
    base_params = filter(lambda p: id(p) not in conv5_params+fuse_params,
                        hed.parameters())

    optimizer = torch.optim.SGD([
                {'params': base_params},
                {'params': hed.moduleVggFiv.parameters(), 'lr': lr * 100},
                {'params': hed.moduleCombine.parameters(), 'lr': lr * 0.001}
                ], lr=lr, momentum=0.9)
    return hed, optimizer
    


def get_hed_model(args):
    modelPath = '/home/agong/HED-on-Cityscapes/model/vgg16.pth'
    hed = models.__dict__['HED']().cuda(args.rank)
    hed.apply(weights_init)
    pretrained_dict = torch.load(modelPath)
    pretrained_dict = convert_vgg(pretrained_dict)

    model_dict = hed.state_dict()
    model_dict.update(pretrained_dict)
    hed.load_state_dict(model_dict)

    lr = 1e-4
    
    fuse_params = list(map(id, hed.fuse.parameters()))
    conv5_params = list(map(id, hed.conv5.parameters()))
    base_params = filter(lambda p: id(p) not in conv5_params+fuse_params,
                        hed.parameters())

    optimizer = torch.optim.SGD([
                {'params': base_params},
                {'params': hed.conv5.parameters(), 'lr': lr * 100},
                {'params': hed.fuse.parameters(), 'lr': lr * 0.001}
                ], lr=lr, momentum=0.9)

    # init

    # if getattr(args, 'ckpt_hed', None) is not None:
    #     args.logger.info('Loading from ckpt %s' % args.ckpt_hed)
    #     ckpt = torch.load(args.ckpt_hed,
    #         map_location=torch.device('cpu'))
    #     if 'hed' in ckpt:
    #         hed.load_state_dict(ckpt['hed'])
    #     if 'optimizer' in ckpt:
    #         optimizer.load_state_dict(ckpt['optimizer'])
    
    return hed, optimizer


class Trainer:

    def __init__(self, args):
        args.logger.info('Initializing trainer')
        if not os.path.isdir('../predict'):
            os.makedirs('../predict')

        torch.cuda.set_device(args.rank)

        self.generator, self.optimG = get_hned_model(args)
        self.generator = torch.nn.parallel.DistributedDataParallel(self.generator, device_ids=[args.rank])
        self.lrDecayEpochs = {3,5,8,10,12,15,18}
        self.gamma = 0.1
        self.BCEloss = nn.BCELoss(reduction='mean')
        self.BCEloss.cuda(args.rank)


        torch.backends.cudnn.benchmark = True
        self.global_step = 0

        if args.resume is not None:
            self.load_checkpoint(args.resume)

        if args.rank == 0:
            self.writer = SummaryWriter(args.path)

        train_dataset, val_dataset = get_dataset(args)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size//args.gpus, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size//args.gpus, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=val_sampler)

        self.args = args
        self.epoch = 0
        self.args.logger.debug('Finish init trainer')

    def set_epoch(self, epoch):
        self.args.logger.info("Start of epoch %d" % (epoch+1))
        self.epoch = epoch + 1
        if self.epoch in self.lrDecayEpochs:
            self.adjustLR()
        self.train_loader.sampler.set_epoch(epoch)
        self.val_loader.sampler.set_epoch(epoch)
        # if self.args.optimizer == 'sgd':
        #     self.lr_scheduler.step(epoch)
        #     if self.args.rank == 0:
        #         self.writer.add_scalar('other/lr-epoch', self.optimizer.param_groups[0]['lr'], self.epoch)

    def train(self):
        self.args.logger.info('Training started')
        self.generator.train()
        end = time()
        for i, (seg, img) in enumerate(self.train_loader):
            load_time = time() -end
            end = time()
            # for tensorboard
            self.global_step += 1

            # forward pass
            seg = seg.cuda(self.args.rank, non_blocking=True)
            img = img.cuda(self.args.rank, non_blocking=True)
            with torch.no_grad():
                seg = torch.clamp((seg*1.1), 0.0, 1.0)
            if random.random() < 0.5:
                with torch.no_grad():
                    seg = torch.flip(seg, [3])
                    img = torch.flip(img, [3])

            d1, d2, d3, d4, d5, d6 = self.generator(img)

            loss1 = self.BCEloss(d1, seg)
            loss2 = self.BCEloss(d2, seg)
            loss3 = self.BCEloss(d3, seg)
            loss4 = self.BCEloss(d4, seg)
            loss5 = self.BCEloss(d5, seg)
            loss6 = self.BCEloss(d6, seg)

            self.loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

            self.optimG.zero_grad()
            self.args.logger.debug(self.loss)
            self.sync([self.loss])
            self.loss.backward()
            self.optimG.step()

            with torch.no_grad():
                dd = d6.clone()
                # self.args.logger.debug(dd.max())
                dd[dd>0.3] = 0.8
                dd[dd<=0.3] = 0.0
            

            comp_time = time() - end
            end = time()

            # print
            if self.args.rank == 0 and i % self.args.print_freq == 0:
                self.args.logger.info(
                    'Epoch [{epoch:d}/{tot_epoch:d}][{cur_batch:d}/{tot_batch:d}] '
                    'load [{load_time:.3f}s] comp [{comp_time:.3f}s] '
                    'loss [{loss:.4f}] '.format(
                        epoch=self.epoch, tot_epoch=self.args.epochs,
                        cur_batch=i+1, tot_batch=len(self.train_loader),
                        load_time=load_time, comp_time=comp_time,
                        loss=self.loss.item()
                    )
                )
                self.writer.add_scalar('train/loss', self.loss.item(), self.global_step)
                self.writer.add_image('train/img', make_grid(img, normalize=True), self.global_step)
                self.writer.add_image('train/edge gt', make_grid(seg, normalize=True), self.global_step)
                self.writer.add_image('train/edge dd', make_grid(dd, normalize=True), self.global_step)
                self.writer.add_image('train/edge', make_grid(d6, normalize=True), self.global_step)


    def validate(self):
        self.args.logger.info('Validation started')
        self.generator.eval()

        val_loss = AverageMeter()

        with torch.no_grad():
            end = time()
            for i, (seg, img) in enumerate(self.val_loader):
                load_time = time()-end
                end = time()

                 # forward pass
                seg = seg.cuda(self.args.rank, non_blocking=True)
                img = img.cuda(self.args.rank, non_blocking=True)
                with torch.no_grad():
                    seg = torch.clamp((seg*1.1), 0.0, 1.0)

                d1, d2, d3, d4, d5, d6 = self.generator(img)

                loss1 = self.BCEloss(d1, seg)
                loss2 = self.BCEloss(d2, seg)
                loss3 = self.BCEloss(d3, seg)
                loss4 = self.BCEloss(d4, seg)
                loss5 = self.BCEloss(d5, seg)
                loss6 = self.BCEloss(d6, seg)

                self.loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
             
                self.args.logger.debug(self.loss)
                # loss and accuracy
                # img.size(0) should be batch size
                size = torch.tensor(float(img.size(0))).cuda(self.args.rank) # pylint: disable=not-callable
                self.loss.mul_(size)
                self.sync([self.loss], mean=False) # sum
                self.loss.div_(size)
                val_loss.update(self.loss.item(), size.item())

                if self.epoch % 1 == 0 and self.args.rank == 0 and i % 100 == 0:
                    p = torch.cat([img.cuda(), seg.cuda(), d6.cuda()], dim=1)
                    p = p.cpu().detach().numpy()
                    np.save('../predict/val_'+str(end)+'_'+str(i).zfill(6)+'.npy', p)

                # if self.epoch % 5 == 0 and self.args.rank == 0 and i % 100 == 0:
                #     self.generate_sequence(frame1, frame2, seg1, seg2)
                
                comp_time = time() - end
                end = time()

                # print
                if self.args.rank == 0 and i % self.args.print_freq == 0:
                    self.args.logger.info(
                        'Epoch [{epoch:d}/{tot_epoch:d}][{cur_batch:d}/{tot_batch:d}] '
                        'load [{load_time:.3f}s] comp [{comp_time:.3f}s]'.format(
                            epoch=self.epoch, tot_epoch=self.args.epochs,
                            cur_batch=i+1, tot_batch=len(self.val_loader),
                            load_time=load_time, comp_time=comp_time,
                        )
                    )

        if self.args.rank == 0:
            self.args.logger.info(
                'Epoch [{epoch:d}/{tot_epoch:d}] loss [{loss:.4f}] '.format(
                    epoch=self.epoch, tot_epoch=self.args.epochs,
                    loss=val_loss.avg,
                )
            )
            self.writer.add_scalar('val/loss', val_loss.avg, self.epoch)

        return {'loss': val_loss.avg}

    def sync(self, tensors, mean=True):
        '''Synchronize all tensors given using mean or sum.'''
        for tensor in tensors:
            dist.all_reduce(tensor)
            if mean:
                tensor.div_(self.args.gpus)

    

    def save_checkpoint(self, metrics):
        self.args.logger.info('Saving checkpoint..')
        prefix = '../checkpoint'
        torch.save({
            'epoch': self.epoch,
            'arch': self.args.arch,
            'generator': self.generator.module.state_dict(), # data parallel
            'optimG': self.optimG.state_dict(),
        }, '%s/%03d.pth' % (prefix, self.epoch))
        shutil.copy('%s/%03d.pth' % (prefix, self.epoch),
            '%s/latest.pth' % prefix)

    def load_checkpoint(self, resume):
        self.args.logger.info('Resuming checkpoint %s' % resume)
        ckpt = torch.load(resume)
        assert ckpt['arch'] == self.args.arch, ('Architecture mismatch: ckpt %s, config %s'
            % (ckpt['arch'], self.args.arch))

        self.epoch = ckpt['epoch']
        self.generator.load_state_dict(ckpt['generator'])
        self.optimG.load_state_dict(ckpt['optimG'])

        self.args.logger.info('Checkpoint loaded')

    def vis_seg_mask(self, seg, n_classes, argmax=False):
        '''
        mask (bs, c,h,w) into normed rgb (bs, 3,h,w)
        all tensors
        '''
        global color_map
        if argmax:
            id_seg = torch.argmax(seg, dim=1)
        else: id_seg = seg
        color_mapp = torch.tensor(color_map)
        rgb_seg = color_mapp[id_seg].permute(0,3,1,2).contiguous().float()
        return rgb_seg/255

    def eval_generate_sequence(self, img1, img2, seg1, seg2):
        seg1 = cv2.imread(seg1, cv2.IMREAD_GRAYSCALE)
        seg2 = cv2.imread(seg2, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(img1)
        img2 = cv2.imread(img2)
        if seg1 is None or seg2 is None or img1 is None or img2 is None:
            self.args.logger.debug('path name not exists')
            return
        seg1 = cv2.resize(seg1, dsize=(256,256), interpolation=cv2.INTER_NEAREST)
        seg2 = cv2.resize(seg2, dsize=(256,256), interpolation=cv2.INTER_NEAREST)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) 
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) 
        to_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
        img1, img2 = to_normalize(img1).unsqueeze_(0), to_normalize(img2).unsqueeze_(0)
        seg1, seg2 = torch.from_numpy(seg1), torch.from_numpy(seg2)
        seg1, seg2 = seg1.float().unsqueeze_(0).unsqueeze_(0), seg2.float().unsqueeze_(0).unsqueeze_(0)
        self.args.logger.debug(seg1.shape)
        self.args.logger.debug(img1.shape)
        self.generate_sequence(img1, img2, seg1, seg2)


    def generate_sequence(self, img1, img2, seg1, seg2):
        img, seg = [], []
        img.append(img1.cuda(self.args.rank, non_blocking=True))
        img.append(img2.cuda(self.args.rank, non_blocking=True))
        seg.append(seg1.cuda(self.args.rank, non_blocking=True))
        seg.append(seg2.cuda(self.args.rank, non_blocking=True))
        with torch.no_grad():
            for i in range(8):
                x = torch.cat([seg[-2], img[-2], img[-1], seg[-1]], dim=1) # zeroth is batch size, first is channel
                x = x.cuda(self.args.rank, non_blocking=True)
                # self.args.logger.debug(x.shape)
                seg_next, img_next = self.netG(x)
                # img_next = F.tanh(img_next)
                img_next = (img_next - self.mean_arr) / self.std_arr
                seg_next = torch.argmax(seg_next, dim=1).unsqueeze_(1).float() # from NCHW to N1HW
                img.append(img_next)
                seg.append(seg_next)
            p = torch.cat(img, dim=1)
            q = torch.cat(seg, dim=1)
            p = p.cpu().detach().numpy()
            q = q.cpu().detach().numpy()
            t = time()
            np.save('../predict/val_'+str(t)+'_'+'img'+'.npy', p)
            np.save('../predict/val_'+str(t)+'_'+'seg'+'.npy', q)
    
    # utility functions to set the learning rate
    def adjustLR(self):
        for param_group in self.optimG.param_groups:
            param_group['lr'] *= self.gamma







