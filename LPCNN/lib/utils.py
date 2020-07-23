import os
import numpy as np
from tqdm import tqdm

from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from lib.dataset.qsm_dataset import QsmDataset

#from lib.model.vdsr.vdsr import VDSR
from lib.model.vdsrr.vdsrr import VDSRR
from lib.model.pdnn.pdnn import PDNN
from lib.model.pldnn.pldnn import PLDNN
from lib.model.pndnn.pndnn import PNDNN
from lib.model.wpdnn.wpdnn import WPDNN
from lib.model.wpldnn.wpldnn import WPLDNN
from lib.model.pudnn.pudnn import PUDNN

from lib.loss.fwl1.fwl1 import fwl1

def prepareDataset(args, root_dir, data_aug=None, normalize=False):
	
	if args.dataset.lower() == 'qsm':

		dataset_path = root_dir + 'qsm_dataset/qsm_B_r/mix5_data/partition/m_partition_data_list1/'
		
		train_dataset = QsmDataset(dataset_path, split='train', number=args.number, tesla=args.tesla, is_norm=normalize)
		val_dataset = QsmDataset(dataset_path, split='validate', number=args.number, tesla=args.tesla, is_norm=normalize)
	
	else:
		raise ValueError('unknown dataset: ' + dataset)

	train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=16, shuffle=True, drop_last=True)
	val_loader = data.DataLoader(val_dataset, batch_size=1, num_workers=16, shuffle=False)

	print('Got {} training examples'.format(len(train_loader.dataset)))
	print('Got {} validation examples'.format(len(val_loader.dataset)))
	
	return train_loader, val_loader

def loadData(args, root_dir, prediction_data, normalize=False):

	if args.dataset == 'qsm':

		prediction_set, subject_num, ori_num, patch_num, case = prediction_data

		if case == 'whole':
			dataset_path = root_dir + 'qsm_dataset/qsm_B_r/mix5_data/whole/m_data_list1/'
		else:
			dataset_path = root_dir + 'qsm_dataset/qsm_B_r/real_data/partition/partition_data_list/'

		if prediction_set == 'train':
			ext_data = QsmDataset(dataset_path, split='train', sep=case, number=args.number, tesla=args.tesla, is_norm=normalize)
		elif prediction_set == 'val':
			ext_data = QsmDataset(dataset_path, split='validate', sep=case, number=args.number, tesla=args.tesla, is_norm=normalize)
		elif prediction_set == 'test':
			ext_data = QsmDataset(dataset_path, split='test', sep=case, number=args.number, tesla=args.tesla, is_norm=normalize)
		elif prediction_set == 'ext':
			ext_data = ext_handle(args, root_dir, prediction_data, normalize, dataset_path)
		else:
			raise ValueError('Unknown extra data category: ' + prediction_set)

	else:
		raise ValueError('unknown dataset: ' + args.dataset)
		
	data_loader = data.DataLoader(ext_data, batch_size=1, num_workers=16, shuffle=False)
	
	print('Got {} testing examples'.format(len(data_loader.dataset)))

	return data_loader

def ext_handle(args, root_dir, prediction_data, normalize, dataset_path):

	prediction_set, subject_num, ori_num, patch_num, case = prediction_data

	input_name = 'ext_phase.txt'
	gt_name = 'ext_gt.txt'
	mask_name = 'ext_mask.txt'
	dk_name = 'ext_wdk.txt'
	#erod_mask_name = 'ext_erod_mask.txt'
	
	temp_sub = subject_num + '/' + ori_num

	if case == 'whole':
		with open(dataset_path + input_name, 'w') as f:
			f.write(root_dir + 'qsm_dataset/qsm_B_r/real_data/whole/phase_data/' + temp_sub + '/' + subject_num + '_' + ori_num + '_LBVSMV.npy\n')
		with open(dataset_path + gt_name, 'w') as f:
			f.write(root_dir + 'qsm_dataset/qsm_B_r/real_data/whole/cosmos_data/' + subject_num + '/' + subject_num + '_cosmos.npy\n')
		with open(dataset_path + mask_name, 'w') as f:
			f.write(root_dir + 'qsm_dataset/qsm_B_r/real_data/whole/mask_data/' + temp_sub + '/' + subject_num + '_' + ori_num + '_mask.npy\n')
		with open(dataset_path + dk_name, 'w') as f:
			f.write(root_dir + 'qsm_dataset/qsm_B_r/real_data/whole/dk_data/' + temp_sub + '/' + subject_num + '_' + ori_num + '_dk.npy\n')
		#with open(dataset_path + erod_mask_name, 'w') as f:
		#	f.write(root_dir + 'hcp_dataset/dti_dataset/whole/erod_mask_data/' + temp_sub + '_erod_mask.npy\n')


	elif case == 'patch':
		with open(dataset_path + input_name, 'w') as f:
			f.write(root_dir + 'qsm_dataset/qsm_B_r/mix_data/partition/phase_pdata' + temp_sub + '/' + subject_num + '_' + ori_num + '_LBVSMV_p' + patch_num + '.npy\n')
		with open(dataset_path + gt_name, 'w') as f:
			f.write(root_dir + 'qsm_dataset/qsm_B_r/mix_data/partition/cosmos_pdata/' + temp_sub + '/' + subject_num + '_' + ori_num + '_cosmos_p' + patch_num + '.npy\n')
		with open(dataset_path + mask_name, 'w') as f:
			f.write(root_dir + 'qsm_dataset/qsm_B_r/mix_data/partition/mask_pdata/' + temp_sub + '/' + subject_num + '_' + ori_num + '_mask_p' + patch_num + '.npy\n')
		with open(dataset_path + ang_name, 'w') as f:
			f.write(root_dir + 'qsm_dataset/qsm_B_r/mix_data/whole/angle_data/' + temp_sub + '/' + subject_num + '_' + ori_num + '_ang.npy\n')
		#with open(dataset_path + erod_mask_name, 'w') as f:
		#	f.write(root_dir + 'hcp_dataset/dti_dataset/partition/erod_mask_pdata/' + temp_sub + '_erod_mask_p' + patch_num + '.npy\n')
			
	else:
		raise ValueError('unknown case: ' + case)

	ext_data = QsmDataset(dataset_path, split='ext', tesla=args.tesla, is_norm=normalize)

	return ext_data

def chooseModel(args, root_dir):
	
	model = None
	if args.model_arch.lower() == 'vdsr':
		model = VDSR()
	elif args.model_arch.lower() == 'vdsrr':
		model = VDSRR()
	elif args.model_arch.lower() == 'unet':
		model = UNET()
	elif args.model_arch.lower() == 'pdnn':
		model = PDNN(root_dir + 'train_gt_mean.npy', root_dir + 'train_gt_std.npy')
	elif args.model_arch.lower() == 'pldnn':
		model = PLDNN(root_dir + 'train_gt_mean.npy', root_dir + 'train_gt_std.npy')
	elif args.model_arch.lower() == 'pndnn':
		model = PNDNN()
	elif args.model_arch.lower() == 'wpdnn':
		model = WPDNN(root_dir + 'train_gt_mean.npy', root_dir + 'train_gt_std.npy')
	elif args.model_arch.lower() == 'wpldnn':
		model = WPLDNN(root_dir + 'train_gt_mean.npy', root_dir + 'train_gt_std.npy')
	elif args.model_arch.lower() == 'pudnn':
		model = PUDNN(root_dir + 'train_gt_mean.npy', root_dir + 'train_gt_std.npy')
	else:
		raise ValueError('Unknown model arch type: ' + args.model_arch.lower())
		
	return model

def chooseLoss(args, option=0):
	
	loss_fn = None
	if args.model_arch.lower() == 'vdsr':
		loss_fn = nn.MSELoss()
	elif args.model_arch.lower() == 'vdsrr' and option == 0:
		loss_fn = nn.MSELoss()
	elif args.model_arch.lower() == 'vdsrr' and option == 1:
		loss_fn = nn.L1Loss()
	elif args.model_arch.lower() == 'unet':
		loss_fn = nn.L1Loss()
	elif args.model_arch.lower() == 'pdnn' and option == 0:
		loss_fn = nn.L1Loss()
	elif args.model_arch.lower() == 'pdnn' and option == 1:
		loss_fn = fwl1
	elif args.model_arch.lower() == 'pldnn' and option == 0:
		loss_fn = nn.L1Loss()
	elif args.model_arch.lower() == 'pldnn' and option == 1:
		loss_fn = fwl1
	elif args.model_arch.lower() == 'pndnn':
		loss_fn = nn.L1Loss()
	elif args.model_arch.lower() == 'wpdnn' and option == 0:
		loss_fn = nn.L1Loss()
	elif args.model_arch.lower() == 'wpdnn' and option == 1:
		loss_fn = fwl1
	elif args.model_arch.lower() == 'wpldnn' and option == 0:
		loss_fn = nn.L1Loss()
	elif args.model_arch.lower() == 'wpldnn' and option == 1:
		loss_fn = fwl1
	elif args.model_arch.lower() == 'pudnn' and option == 0:
		loss_fn = nn.L1Loss()
	elif args.model_arch.lower() == 'pudnn' and option == 1:
		loss_fn = fwl1
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

	path = root_dir + 'qsm_dataset/qsm_B_r/real_data/whole/data_list/'
	max_val = np.load(path + 'train_val_gt_max_val.npy')
	min_val = np.load(path + 'train_val_gt_min_val.npy')

	mod_input = np.copy(input_data)
	mod_input[mod_input < min_val] = min_val
	mod_input[mod_input > max_val] = max_val
	
	if roi:
		psnr_value = psnr(gt[mask==1], mod_input[mask==1], max_val - min_val)
	else:
		psnr_value = psnr(gt, mod_input, max_val - min_val)
	
	return psnr_value

def qsm_ssim(gt, input_data, mask, root_dir):

	path = root_dir + 'qsm_dataset/qsm_B_r/real_data/whole/data_list/'
	max_val = np.load(path + 'train_val_gt_max_val.npy')
	min_val = np.load(path + 'train_val_gt_min_val.npy')
	
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

