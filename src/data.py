import numpy as np
import os
import sys
import pickle
import json
import time
import glob
from scipy.misc import imread

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from folder import *
# from cityscape_utils import *
'''
	input:			 dataset name(str)

	return np data:	
		n_classes	: int

		train_imgs	: (n_t, h, w, 3)
		train_segs	: (n_t, h, w)
		train_masks	: (n_t, h, w)  missing region is 0, known region is 1 

		val_imgs	: (n_v, h, w, 3)
		val_segs	: (n_v, h, w)
		val_masks	: (n_v, h, w)
'''
def get_dataset(args):
	### explicitly set flip = True #######
	if args.dataset == "cityscape":
		train_dataset = ImageFolder(
			args.train_dir,
			transform=transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
			]),
			target_transform = transforms.Compose([
				transforms.ToTensor(),
			])
		)
		val_dataset = ImageFolder(
			args.val_dir,
			transform=transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
			]),
			target_transform = transforms.Compose([
				transforms.ToTensor(),
			])
		)
		# test_dataset = datasets.ImageFolder(
		# 	args.test_dir,
		# 	transforms.Compose([
		# 		transforms.ToTensor(),
		# 	])
		# )
	else:
		assert False, 'Invalid dataset %s' % args.data
	
	return train_dataset, val_dataset


