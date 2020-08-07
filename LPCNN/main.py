import os
import sys
from pathlib import Path
import argparse
import numpy as np

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.utils import data

from lib.utils import *
from lib.tool.tool import *

import timeit
import socket

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# output directory
root_dir = Path('data/')
checkpoint_dir = Path('checkpoints/')
tb_log_dir = Path('LPCNN/tb_log/')
vis_output_path = Path('LPCNN/result/')

# parameter for relative value evaluation(validation)
validation_mse = np.load(str(root_dir / Path('numpy_data/partition/list/validation_mse.npy')))
whole_validation_mse = np.load(str(root_dir / Path('numpy_data/whole/list/validation_mse.npy')))

def main(args):

	device = torch.device('cuda:0' if not args.no_cuda else 'cpu')

	if args.mode == 'train':

		## experiment name
		exp_name = args.model_arch + args.name

		## tensorboard log
		tb_writer = SummaryWriter(str(tb_log_dir / exp_name))

		## data augmentation
		# Todo :design which type of data aug
		data_aug = None

		## load dataset
		print('Load dataset...')
		train_loader, val_loader = prepareDataset(args, root_dir, data_aug, normalize=args.norm)
		print(args.dataset.lower() + ' dataset loaded.')

		## load model
		model = chooseModel(args, train_loader.dataset.root_path)
		model.to(device)
		print(args.model_arch + ' loaded.')

		## parallel model
		if args.gpu_num > 1:
			model = nn.DataParallel(model)

		## loss function and optimizer
		loss_fn = chooseLoss(args, 0)
		optimizer = chooseOptimizer(model, args)
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.8)

		## initailize statistic result
		start_epoch = 0
		best_per_index = 1000
		total_tb_it = 0

		## resume training
		if not args.resume_file == None:
			start_epoch = torch.load(args.resume_file, map_location=device)['epoch'] + 1
			model.load_state_dict(torch.load(args.resume_file, map_location=device)['model_state'])
			model.to(device)
			print(args.resume_file + ' loaded.')
		
			optimizer.load_state_dict(torch.load(args.resume_file, map_location=device)['optimizer_state'])	

		## train
		for epoch in range(start_epoch, args.num_epoch):
			
			total_tb_it = train(args, device, model, train_loader, epoch, loss_fn, optimizer, tb_writer, total_tb_it)
			mse_index = validation(device, model, val_loader, epoch, loss_fn, tb_writer)
			for param_group in optimizer.param_groups:
				print('Learning rate: %.8f' %(param_group['lr']))	

			scheduler.step(epoch)
			
			state = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}

			if mse_index <= best_per_index:
				best_per_index = mse_index
				best_name = str(checkpoint_dir / Path(exp_name +'_Bmodel.pkl'))
				torch.save(state, best_name)
			else:
				name = str(checkpoint_dir / Path(exp_name +'_Emodel.pkl'))
				torch.save(state, name)
			
		tb_writer.close()

	elif args.mode == 'predict':

		print('Load data...')
		data_loader = loadData(args, root_dir, (args.pred_set, args.size), normalize=args.norm)

		## load model
		model = chooseModel(args, data_loader.dataset.root_path)
		model.load_state_dict(torch.load(args.resume_file, map_location=device)['model_state'])
		model.to(device)
		print(sum([p.numel() for p in model.parameters()]))
		print(args.model_arch + ' loaded.')

		## parallel model
		if args.gpu_num > 1:
			model = nn.DataParallel(model)

		model_name = Path(args.resume_file).parts[-1].split('.')[0]
		print(model_name)

		## loss function and optimizer
		loss_fn = chooseLoss(args, 0)
		optimizer = chooseOptimizer(model, args)
		#optimizer.load_state_dict(torch.load(args.resume_file, map_location=device)['optimizer_state'])

		predict(args, device, model, data_loader, loss_fn, model_name)
	else:
		raise Exception('Unrecognized mode.')

def train(args, device, model, train_loader, epoch, loss_fn, optimizer, tb_writer, total_tb_it):

	print_freq = (len(train_loader.dataset) // args.batch_size) // 3

	model.train()

	for batch_count, (phase_data_list, gt_data, mask_data, dipole_data_list, input_name) in enumerate(train_loader):

		#cuda
		phase_data_list = phase_data_list.to(device, dtype=torch.float)
		gt_data = gt_data.to(device, dtype=torch.float)
		mask_data = mask_data.to(device, dtype=torch.float)

		optimizer.zero_grad()

		output_data = model(phase_data_list, dipole_data_list, mask_data.unsqueeze(1))

		loss = loss_fn(output_data, gt_data)

		loss.backward()

		optimizer.step()

		per_loss = loss.item()

		tb_writer.add_scalar(str(Path('train/overall_loss')), per_loss, total_tb_it)
		total_tb_it += 1

		if batch_count%print_freq == 0:
			print('Epoch [%d/%d] Loss: %.8f' %(epoch, args.num_epoch, per_loss))

	return total_tb_it

def validation(device, model, val_loader, epoch, loss_fn, tb_writer):

	model.eval()

	tb_loss = 0
	mse_loss = 0
	ssim_perf = 0
	psnr_perf = 0

	with torch.no_grad():

		for batch_count, (phase_data_list, gt_data, mask_data, dipole_data_list, input_name) in enumerate(val_loader):

			#cuda
			phase_data_list = phase_data_list.to(device, dtype=torch.float)
			gt_data = gt_data.to(device, dtype=torch.float)
			mask_data = mask_data.to(device, dtype=torch.float)

			output_data = model(phase_data, dipole_data_list, mask_data.unsqueeze(1))

			loss = loss_fn(output_data, gt_data)

			tb_loss += loss.item()

			mask = torch.squeeze(mask_data, 0).cpu().numpy()[:,:,:,np.newaxis]
			
			if args.norm:
				og_output = torch.squeeze(output_data, 0).permute(1, 2, 3, 0).cpu().numpy() * val_loader.dataset.gt_std
				og_output = og_output + val_loader.dataset.gt_mean
				og_output = og_output * mask

				og_gt = torch.squeeze(gt_data, 0).permute(1, 2, 3, 0).cpu().numpy() * val_loader.dataset.gt_std
				og_gt = og_gt + val_loader.dataset.gt_mean
				og_gt = og_gt * mask

			else:
				og_output = torch.squeeze(output_data, 0).permute(1, 2, 3, 0).cpu().numpy() * mask
				og_gt = torch.squeeze(gt_data, 0).permute(1, 2, 3, 0).cpu().numpy() * mask

			mse_loss += np.sqrt(qsm_mse(og_gt, og_output, mask, roi=True))
			psnr_perf += qsm_psnr(og_gt, og_output, mask, root_dir, roi=True)
			ssim_perf += qsm_ssim(og_gt, og_output, mask, root_dir)

		avg_tb_loss = tb_loss / len(val_loader.dataset)
		avg_mse_loss = mse_loss / len(val_loader.dataset) / np.sqrt(validation_mse)
		avg_psnr_perf = psnr_perf / len(val_loader.dataset)
		avg_ssim_perf = ssim_perf / len(val_loader.dataset)
		print('alpha: %.3f' %(model.alpha.cpu().numpy()))
		print('##Validation loss: %.8f RMSE: %.8f PSNR: %.8f SSIM: %.8f' %(avg_tb_loss, avg_mse_loss, avg_psnr_perf, avg_ssim_perf))

		tb_writer.add_scalar(str(Path('val/overall_loss')), avg_tb_loss, epoch)
		tb_writer.add_scalar(str(Path('val/RMSE')), avg_mse_loss, epoch)
		tb_writer.add_scalar(str(Path('val/PSNR')), avg_psnr_perf, epoch)
		tb_writer.add_scalar(str(Path('val/SSIM')), avg_ssim_perf, epoch)

	return avg_mse_loss

def predict(args, device, model, data_loader, loss_fn, model_name):

	nifti_path = root_dir / 'nifti_data'

	model.eval()

	mse_loss = 0
	ssim_perf = 0
	psnr_perf = 0

	with torch.no_grad():
		for batch_count, (phase_data_list, gt_data, mask_data, dipole_data_list, input_name) in enumerate(data_loader):

			#cuda
			phase_data_list = phase_data_list.to(device, dtype=torch.float)
			gt_data = gt_data.to(device, dtype=torch.float)
			mask_data = mask_data.to(device, dtype=torch.float)

			output_data = model(phase_data_list, dipole_data_list, mask_data.unsqueeze(1))
				
			mask = torch.squeeze(mask_data, 0).cpu().numpy()[:,:,:,np.newaxis]

			if args.norm:
				og_output = torch.squeeze(output_data, 0).permute(1, 2, 3, 0).cpu().numpy() * data_loader.dataset.gt_std
				og_output = og_output + data_loader.dataset.gt_mean
				og_output = og_output * mask

				og_gt = torch.squeeze(gt_data, 0).permute(1, 2, 3, 0).cpu().numpy() * data_loader.dataset.gt_std
				og_gt = og_gt + data_loader.dataset.gt_mean
				og_gt = og_gt * mask

			else:
				og_output = torch.squeeze(output_data, 0).permute(1, 2, 3, 0).cpu().numpy() * mask
				og_gt = torch.squeeze(gt_data, 0).permute(1, 2, 3, 0).cpu().numpy() * mask

			mse_loss += np.sqrt(qsm_mse(og_gt, og_output, mask, roi=True))
			ssim_perf += qsm_ssim(og_gt, og_output, mask, root_dir)	
			psnr_perf += qsm_psnr(og_gt, og_output, mask, root_dir, roi=True)	
			#print('##Test PSNR: %.8f SSIM: %.8f' %(qsm_psnr(og_gt, og_output, mask, root_dir, roi=True), qsm_ssim(og_gt, og_output, mask, root_dir)))

			if not args.no_save:

				subject = input_name[0].split('_')[0]
				ori = input_name[0].split('_')[1]
	
				qsm_path = nifti_path / subject / 'cosmos' / (subject + '_cosmos.nii.gz')

				output_path = vis_output_path / (model_name  + '_num' + str(args.number))
				output_path.mkdir(parents=True, exist_ok=True)

				if args.size == 'whole':
					save_name = vis_output_path / (model_name + '_num' + str(args.number)) / (subject + '_' + ori)
				else:
					patch_num = input_name[0].split('_')[2] 
					save_name = vis_output_path / (model_name + '_num' + str(args.number)) / (subject + '_' + ori + '_' + patch_num)

				qsm_display(og_output, str(qsm_path), torch.squeeze(mask_data, 0).cpu().numpy(), out_name=str(save_name))

		avg_mse_loss = mse_loss / len(data_loader.dataset) / np.sqrt(validation_mse)
		avg_ssim_perf = ssim_perf / len(data_loader.dataset)
		avg_psnr_perf = psnr_perf / len(data_loader.dataset)

		print('##Test RMSE: %.8f PSNR: %.8f SSIM: %.8f' %(avg_mse_loss, avg_psnr_perf, avg_ssim_perf))


parser = argparse.ArgumentParser(description='QSM Inversion problem')
parser.add_argument('--mode', default='train', choices=['train', 'predict'], help='operation mode: train or predict (default: train)')
parser.add_argument('--name', type=str, default='_test', help='the name of experiment')
parser.add_argument('--number', type=int, default=3, choices=[1, 2, 3], help='input phase number')
parser.add_argument('--dataset', default='qsm', choices=['qsm'], help='dataset to use (default: qsm)')
parser.add_argument('--norm', action='store_true', default=True, help='normalize data')
parser.add_argument('--tesla', default=7, type=int, choices=[3, 7], help='B0 tesla(default: 7)')
parser.add_argument('--gpu_num', default=1, type=int, choices=[1, 2, 3, 4], help='number of gpu (default: 1)')
parser.add_argument('--model_arch', default='lpcnn', choices=['lpcnn'], help='network architecture (default: lpcnn)')
parser.add_argument('--num_epoch', default=100, type=int, metavar='N', help='number of total epochs to run (default: 100)')
parser.add_argument('--batch_size', type=int, default=2, help='batch size (default: 2)')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam'], help='optimizer to use (default: adam)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--no_save', action='store_true', default=False, help='disables saving result')
parser.add_argument('--resume_file', type=str, default=None, help='the checkpoint file to resume from')
parser.add_argument('--pred_set', default='val', choices=['train', 'val', 'test'], help='choose which dataset partition for prediction')
parser.add_argument('--size', default='whole', choices=['whole', 'patch'],  help='choose whole or patch images for prediction')

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)
