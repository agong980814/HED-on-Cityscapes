import argparse
import datetime
import logging
import pathlib
import random
import socket
import sys

import torch
import torch.distributed as dist 
import torch.multiprocessing as mp 
import numpy as np 

from trainer import Trainer 

def get_exp_path():
	'''Retrun new experiment path.'''
	return '../log/exp-{0}'.format(
		datetime.datetime.now().strftime('%m-%d-%H:%M:%S'))


def get_logger(path, rank=None):
	'''Get logger for experiment.'''
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.DEBUG)

	if rank is None:
		formatter = logging.Formatter('%(asctime)s-%(message)s')
	else:
		formatter = logging.Formatter('%(asctime)s - [worker '
			+ str(rank) +'] - %(message)s')
	
	# stderr log
	handler = logging.StreamHandler(sys.stderr)
	handler.setLevel(logging.DEBUG)
	handler.setFormatter(formatter)
	logger.addHandler(handler)

	# file log
	handler = logging.FileHandler(path)
	handler.setLevel(logging.DEBUG)
	handler.setFormatter(formatter)
	logger.addHandler(handler)

	return logger


def worker(rank, args):
	logger = get_logger(args.path + '/experiment.log',
						rank) # process specific logger
	args.logger = logger
	args.rank = rank
	dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:%d' % args.port,
		world_size=args.gpus, rank=args.rank)

	# seed
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	trainer = Trainer(args)

	if args.img1 is not None and args.img2 is not None and args.seg1 is not None and args.seg2 is not None:
		trainer.model.eval()
		trainer.eval_generate_sequence(args.img1, args.img2, args.seg1, args.seg2)
		return
	
	if args.validate:
  		trainer.validate()
	else:
		if args.rank == 0:
			pathlib.Path('../checkpoint').mkdir(
					parents=True, exist_ok=False)

		for epoch in range(args.epochs):
			trainer.set_epoch(epoch)
			trainer.train()
			metrics = trainer.validate()

			if args.rank == 0:	# gpu id
				trainer.save_checkpoint(metrics)


def main():
	parser = argparse.ArgumentParser(description='Train a segmentation completion network')
	parser.add_argument('-d', '--dataset', type=str, default='cityscape',
						help='training dataset', choices=['cityscape'])
	parser.add_argument('--train_dir', type=str,
						default='/data/agong/HED/train', help='Cityscape train dir')
	parser.add_argument('--val_dir', type=str,
						default='/data/agong/HED/val', help='Cityscape val dir')
	parser.add_argument('--test_dir', type=str,
						default='/data/agong/test', help='Cityscape test dir')
	parser.add_argument('--validate', action='store_true',
						help='whether eval after each training')
	parser.add_argument('--val_interval', dest='val_interval',
						help='number of epochs to evaluate',type=int,default=1)
	parser.add_argument('-a', '--arch', type=str, default='HED', help='model to use',
						choices=['GridNet','CoordGridNet','ResnetGenerator','HED'])
	parser.add_argument('--discriminator', type=str, default='NLayerDiscriminator', help='model to use')
	parser.add_argument('--generator', type=str, default='ResnetGenerator', help='model to use')
	parser.add_argument('-bs','--batch_size', type=int,
						default=32, help='Batch size (over multiple gpu)')
	parser.add_argument('-e', '--epochs', type=int,
						default=20, help='Number of training epochs')
	parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
	parser.add_argument('--img1', type=str, default=None, help='First image url')
	parser.add_argument('--img2', type=str, default=None, help='Second image url')
	parser.add_argument('--seg1', type=str, default=None, help='First image seg url')
	parser.add_argument('--seg2', type=str, default=None, help='Second image seg url')
	# parser.add_argument('-emb', '--embedding_dim', type=int,
	# 					default=15, help="embedding dimension")
	# idstributed training
	parser.add_argument('-j', '--workers', type=int, default=4,
						help='Number of data loading workers')
	parser.add_argument('--port', type=int, default=None, help='Port for distributed training')
	parser.add_argument('--seed', type=int, default=1024, help='Random seed')
	parser.add_argument('--print_freq', type=int,
						default=10, help='Print frequency')
	# save and load
	parser.add_argument('-p', '--path', type=str,
						default=None, help='Experiment path')
	parser.add_argument('--ckpt', type=str, default=None,
						help='Path to checkpoint to load')
	# Lin
	parser.add_argument('--start_epoch', dest='start_epoch',
											help='starting epoch',
											default=1, type=int)
	parser.add_argument('--disp_interval', dest='disp_interval',
											help='number of iterations to display',
											default=10, type=int)
	# config optimization
	parser.add_argument('--o', dest='optimizer', help='training optimizer',
						choices =['adamax','adam', 'sgd'], default="adamax")
	parser.add_argument('--lr', dest='lr', help='starting learning rate',
						default=0.0002, type=float)
	parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
	parser.add_argument('--lr_decay_step', dest='lr_decay_step', help='step to do learning rate decay, unit is epoch',
						default=5, type=int)
	parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
						help='learning rate decay ratio', default=0.1, type=float)

	parser.add_argument('--input_nc', type=int, default=8, help='# of input image channels: 3 for RGB and 1 for grayscale')
	parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
	parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
	parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
	parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
	parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
	parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
	parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
	parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
	parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
	parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
	parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
	
	args = parser.parse_args()
	
	# exp path
	if args.path is None:
		args.path = get_exp_path()
	pathlib.Path(args.path).mkdir(parents=True, exist_ok=False)
	(pathlib.Path(args.path) / 'checkpoint').mkdir(parents=True, exist_ok=False)

	# find free port
	if args.port is None:
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			s.bind(('', 0))
			s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			args.port = int(s.getsockname()[1])

	# logger
	logger = get_logger(args.path + '/experiment.log')
	logger.info('Start of experiment')
	logger.info('=========== Initilized logger =============')
	logger.info('\n\t' + '\n\t'.join('%s: %s' % (k, str(v))
		for k, v in sorted(dict(vars(args)).items())))
	
	# distributed training
	args.gpus = torch.cuda.device_count()
	logger.info('Total number of gpus: %d' % args.gpus)
	mp.spawn(worker, args=(args,), nprocs=args.gpus)

if __name__ == '__main__':
	main()
