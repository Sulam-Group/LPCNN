import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from lib.dataset.qsm_dataset import QsmDataset

from lib.model.lpcnn.lpcnn import LPCNN


def prepareDataset(args, root_dir, data_aug=None, normalize=False):
	
	if args.dataset.lower() == 'qsm':

		dataset_path = root_dir / 'numpy_data'
		
		train_dataset = QsmDataset(dataset_path, split='train', number=args.number, tesla=args.tesla, is_norm=normalize)
		val_dataset = QsmDataset(dataset_path, split='validation', number=args.number, tesla=args.tesla, is_norm=normalize)
	
	else:
		raise ValueError('unknown dataset: ' + dataset)

	train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=16, shuffle=True, drop_last=True)
	val_loader = data.DataLoader(val_dataset, batch_size=1, num_workers=16, shuffle=False)

	print('Got {} training examples'.format(len(train_loader.dataset)))
	print('Got {} validation examples'.format(len(val_loader.dataset)))
	
	return train_loader, val_loader

def loadData(args, root_dir, prediction_data, normalize=False):
	
	if args.dataset == 'qsm':
		
		prediction_set, case = prediction_data

		if case == 'whole':
			dataset_path = root_dir / 'numpy_data'
		else:
			dataset_path = root_dir / 'numpy_data'

		if prediction_set == 'train':
			ext_data = QsmDataset(dataset_path, split='train', sep=case, number=args.number, tesla=args.tesla, is_norm=normalize)
		elif prediction_set == 'val':
			ext_data = QsmDataset(dataset_path, split='validation', sep=case, number=args.number, tesla=args.tesla, is_norm=normalize)
		elif prediction_set == 'test':
			ext_data = QsmDataset(dataset_path, split='test', sep=case, number=args.number, tesla=args.tesla, is_norm=normalize)
		else:
			raise ValueError('Unknown extra data category: ' + prediction_set)

	else:
		raise ValueError('unknown dataset: ' + args.dataset)

	data_loader = data.DataLoader(ext_data, batch_size=1, num_workers=16, shuffle=False)

	print('Got {} testing examples'.format(len(data_loader.dataset)))

	return data_loader

def chooseModel(args, root_dir):
	
	model = None
	if args.model_arch.lower() == 'lpcnn':
		model = LPCNN(root_dir / 'train_gt_mean.npy', root_dir / 'train_gt_std.npy')
	else:
		raise ValueError('Unknown model arch type: ' + args.model_arch.lower())
		
	return model

def chooseLoss(args, option=0):
	
	loss_fn = None
	if args.model_arch.lower() == 'lpcnn' and option == 0:
		loss_fn = nn.MSELoss()
	elif args.model_arch.lower() == 'lpcnn' and option == 1:
		loss_fn = nn.L1Loss()
	else:
		raise ValueError('Unsupported loss function')
		
	return loss_fn

def chooseOptimizer(model, args):
	
	optimizer = None
	if args.optimizer == 'sgd':
		optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
	elif args.optimizer == 'adam':
		optimizer = optim.Adam(model.parameters(), args.learning_rate, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
	elif args.optimizer == 'custom':
		pass
	else:
		raise ValueError('Unsupported optimizer: ' + args.optimizer)

	return optimizer
	
def qsm_psnr(gt, input_data, mask, root_dir, roi=True):

	path = root_dir / Path('numpy_data/whole/list/')
	max_val = np.load(str(path / 'train_val_gt_max.npy'))
	min_val = np.load(str(path / 'train_val_gt_min.npy'))

	mod_input = np.copy(input_data)
	mod_input[mod_input < min_val] = min_val
	mod_input[mod_input > max_val] = max_val
	
	if roi:
		psnr_value = psnr(gt[mask==1], mod_input[mask==1], max_val - min_val)
	else:
		psnr_value = psnr(gt, mod_input, max_val - min_val)
	
	return psnr_value

def qsm_ssim(gt, input_data, mask, root_dir):

	path = root_dir / Path('numpy_data/whole/list/')
	max_val = np.load(str(path / 'train_val_gt_max.npy'))
	min_val = np.load(str(path / 'train_val_gt_min.npy'))
	
	mod_input = np.copy(input_data)
	
	mod_input[mod_input < min_val] = min_val
	mod_input[mod_input > max_val] = max_val

	new_gt = (gt - min_val) / (max_val - min_val)
	new_input = (mod_input - min_val) / (max_val - min_val)
	
	ssim_value = ssim(new_gt, new_input, multichannel=True, data_range=1, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

	return ssim_value

def qsm_mse(gt, input_data, mask, roi=True):

	if roi:
		total = np.sum(mask)
	else:
		(x, y, z, _) = gt.shape
		total = x * y * z
	
	mse = np.sum(np.square(gt - input_data)) / total

	return mse

