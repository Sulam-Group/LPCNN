import os
import sys
import argparse
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.utils import data

from lib.utils import *
from tool.tool import qsm_display

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

data_normalization = True
test_mean_std = False

gamma = 42.57747892

# parameter for relative value evaluation(validate)
## mix_data
whole_validate_mse = 0.0007885226196501292

## real_data
#whole_validate_mse = 0.0007881615545795491

# output directory
root_dir = '/home/kuowei/Desktop/'
vis_output_path = './'

input_data_std_path = '/home/kuowei/Desktop/qsm_dataset/qsm_B_r/mix_data/whole/data_list/train_input_std.npy'
input_data_mean_path = '/home/kuowei/Desktop/qsm_dataset/qsm_B_r/mix_data/whole/data_list/train_input_mean.npy'

gt_data_std_path = '/home/kuowei/Desktop/qsm_dataset/qsm_B_r/mix_data/whole/data_list/train_gt_std.npy'
gt_data_mean_path = '/home/kuowei/Desktop/qsm_dataset/qsm_B_r/mix_data/whole/data_list/train_gt_mean.npy'

def main(args):
	
	device = torch.device('cuda:0' if not args.no_cuda else 'cpu')

	# naming
	example_name = args.model_arch + args.save_name

	# load data
	phase_data = nib.load(args.phase_file)
	phase_numpy = phase_data.get_fdata() / (args.tesla * gamma)

	ang_data = np.load(args.ang_file)

	mask_data = nib.load(args.mask_file)
	mask_numpy = mask_data.get_fdata()

	if not args.gt_file == None:
		gt_data = nib.load(args.gt_file)
		gt_numpy = gt_data.get_fdata()

	if test_mean_std:
		sq_data_sum = np.sum(np.square(phase_numpy[mask_numpy==1]))
		data_sum = np.sum(phase_numpy[mask_numpy==1])
		voxel_num = mask_numpy.sum()
		
		input_data_mean = data_sum / voxel_num
		input_data_std = np.sqrt((sq_data_sum - voxel_num * np.square(input_data_mean)) / voxel_num)
	else:
	
		input_data_std = np.load(input_data_std_path)
		input_data_mean = np.load(input_data_mean_path)

	
	gt_data_std = np.load(gt_data_std_path)
	gt_data_mean = np.load(gt_data_mean_path)

	# normalization
	phase_numpy = phase_numpy - input_data_mean
	phase_numpy = phase_numpy / input_data_std
	phase_numpy = phase_numpy * mask_numpy

	mask = mask_numpy[:,:,:,np.newaxis]

	# load model
	model = chooseModel(args)
	model.load_state_dict(torch.load(args.resume_file, map_location=device)['model_state'])
	model.to(device)
	print(args.model_arch + ' loaded.')
	
	# parallel model
	if args.gpu_num > 1:
		model = nn.DataParallel(model)

	model_name = args.resume_file.split('/')[-1].split('.')[0]
	print(model_name)

	mse_loss = 0
	ssim_perf = 0
	w_psnr_perf = 0
	r_psnr_perf = 0

	model.eval()

	with torch.no_grad():
	
		#cuda
		phase_numpy = torch.from_numpy(phase_numpy[np.newaxis,:,:,:][np.newaxis,:,:,:,:])
		phase_numpy = phase_numpy.to(device, dtype=torch.float)

		ang_data = torch.from_numpy(ang_data[np.newaxis,:])
		ang_data = ang_data.to(device, dtype=torch.float)
	
		output_data = model(phase_numpy, ang_data)

		#reverse norm	
		og_output = torch.squeeze(output_data, 0).permute(1, 2, 3, 0).cpu().numpy() * gt_data_std
		og_output = og_output + gt_data_mean
		og_output = og_output * mask

		
		if not args.gt_file == None:
			og_gt = gt_numpy[:, :, :, np.newaxis] * mask
			mse_loss += qsm_mse(og_gt, og_output, mask, roi=True)
			ssim_perf += qsm_ssim(og_gt, og_output, mask, root_dir)	
			w_psnr_perf += qsm_psnr(og_gt, og_output, mask, root_dir, roi=False)
			r_psnr_perf += qsm_psnr(og_gt, og_output, mask, root_dir, roi=True)

		if not os.path.exists(vis_output_path + model_name):
			os.makedirs(vis_output_path + model_name)

		save_name = vis_output_path + model_name + '/' + example_name

		qsm_display(og_output, args.phase_file, mask_numpy, out_name=save_name)

	if not args.gt_file == None:
		avg_mse_loss = mse_loss / whole_validate_mse
		avg_ssim_perf = ssim_perf
		avg_w_psnr_perf = w_psnr_perf
		avg_r_psnr_perf = r_psnr_perf

		print('##Test Mse: %.8f PSNR: %.8f(%.8f) SSIM: %.8f' %(avg_mse_loss, avg_w_psnr_perf, avg_r_psnr_perf, avg_ssim_perf))

parser = argparse.ArgumentParser(description='QSM Inference')
parser.add_argument('--save_name', type=str, default='_example', help='the name of testing example')
parser.add_argument('--phase_file', type=str, default='example.nii.gz', help='the testing qsm data')
parser.add_argument('--tesla', default=7, type=int, choices=[3, 7], help='B0 tesla(default: 7)')
parser.add_argument('--ang_file', type=str, default='ang.npy', help='the testing qsm ang data')
parser.add_argument('--mask_file', type=str, default='mask.nii.gz', help='the testing mask data')
parser.add_argument('--gt_file', type=str, default=None, help='the testing gt')
parser.add_argument('--gpu_num', default=1, type=int, choices=[1, 2, 3, 4], help='number of gpu (default: 1)')
parser.add_argument('--model_arch', default='vdsr', choices=['edsr', 'srcnn', 'vdsr', 'vdsrr', 'espcn', 'wvdsrr', 'mvdsrr', 'mvdsrrv2', 'mvdsrrbn', 'mvdsrrbn2', 'mwvdsrrbn', 'mwdvdsrrbn'], help='network model (default: edsr)')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--no_save', action='store_true', default=False, help='disables saving tensors')
parser.add_argument('--resume_file', type=str, default='./checkpoint/vdsr_4xus_exp00_18_64_bs_2_lr_1e-5_test40_model.pkl', help='the checkpoint file to resume from')

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)
