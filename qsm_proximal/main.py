import os
import sys
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
from tool.tool import *

import timeit
import socket

data_normalization = True

# parameter for relative value evaluation(validate)
## mix16_data
#validate_mse = 0.0006881151174021742
#whole_validate_mse = 0.0008006943643658826 
## synthetic500_data
#validate_mse = 0.0006969514815906405
#whole_validate_mse = 0.0008042192171182903
## synthetic100_data
#validate_mse = 0.0006935200203605111 #partition100
#validate_mse = 0.0006913588246836717 #partition
#whole_validate_mse = 0.0008042192171182896
## mix_data
#validate_mse = 0.0006488927369219911
#whole_validate_mse = 0.0007885226196501292
## real_data
validate_mse = 0.0006488490366104272
whole_validate_mse = 0.0007881615545795491

# output directory
if socket.gethostname() == 'ka':
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	root_dir = '/home/kuowei/Desktop/'
elif socket.gethostname() == 'rafiki':
	os.environ['CUDA_VISIBLE_DEVICES'] = '1'
	root_dir = '/mnt/data0/kuowei/'
elif socket.gethostname() == 'kuowei':
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	root_dir = '/home/kuowei/Desktop/'
else:
	os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
	root_dir = '/home-3/klai10@jhu.edu/data/kuowei/'
checkpoint_dir = './checkpoint/'
output_log_dir = './log/'
tb_log_dir = './tb_log/'
vis_output_path = './vis_output/'

# prediction data number
prediction_set = 'val' #['train', 'val', 'test', 'ext']
subject_num = 'Sub003'
ori_num = 'ori1'
case = 'whole' #['patch', 'whole']
patch_num = '0'

prediction_data = (prediction_set, subject_num, ori_num, patch_num, case)

def main(args):
	start = timeit.default_timer()
	device = torch.device('cuda:0' if not args.no_cuda else 'cpu')

	if args.mode == 'train':

		## experiment name
		exp_name = args.model_arch + args.name

		## tensorboard log
		tb_writer = SummaryWriter(tb_log_dir + exp_name)

		## data augmentation
		# Todo :design which type of data aug
		data_aug = None

		## load dataset
		print('Load dataset...')
		train_loader, val_loader = prepareDataset(args, root_dir, data_aug, normalize=data_normalization)
		print(args.dataset.lower() + ' dataset loaded.')

		## load model
		model = chooseModel(args, train_loader.dataset.root)
		model.to(device)
		print(args.model_arch + ' loaded.')

		## parallel model
		if args.gpu_num > 1:
			model = nn.DataParallel(model)

		## loss function and optimizer
		loss_fn = chooseLoss(args, 2)
		optimizer = chooseOptimizer(model, args)
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.8)
		#scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
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

		for epoch in range(start_epoch, args.num_epoch):
			
			total_tb_it = train(args, device, model, train_loader, epoch, loss_fn, optimizer, tb_writer, total_tb_it)
			mse_index = validate(device, model, val_loader, epoch, loss_fn, tb_writer)
			for param_group in optimizer.param_groups:
				print('Learning rate: %.8f' %(param_group['lr']))	

			scheduler.step(epoch)
			
			state = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}

			if mse_index <= best_per_index:
				best_per_index = mse_index
				best_name = checkpoint_dir + exp_name +'_Bmodel.pkl'
				torch.save(state, best_name)
			else:
				name = checkpoint_dir + exp_name +'_Emodel.pkl'
				torch.save(state, name)
			
		tb_writer.close()

	elif args.mode == 'predict':

		print('Load data...')
		data_loader = loadData(args, root_dir, prediction_data, normalize=data_normalization)

		## load model
		model = chooseModel(args, data_loader.dataset.root)
		model.load_state_dict(torch.load(args.resume_file, map_location=device)['model_state'])
		model.to(device)
		print(sum([p.numel() for p in model.parameters()]))
		print(args.model_arch + ' loaded.')

		## parallel model
		if args.gpu_num > 1:
			model = nn.DataParallel(model)

		model_name = args.resume_file.split('/')[-1].split('.')[0]
		print(model_name)

		## loss function and optimizer
		loss_fn = chooseLoss(args, 1)
		optimizer = chooseOptimizer(model, args)
		#optimizer.load_state_dict(torch.load(args.resume_file, map_location=device)['optimizer_state'])

		predict(args, device, model, data_loader, loss_fn, model_name)
	else:
		raise Exception('Unrecognized mode.')
	stop = timeit.default_timer()
	print('Time: ', stop - start)

def train(args, device, model, train_loader, epoch, loss_fn, optimizer, tb_writer, total_tb_it):

	print_freq = (len(train_loader.dataset) // args.batch_size) // 3

	model.train()

	for batch_count, (input_data, gt_data, mask_data, dk_data, input_name) in enumerate(train_loader):

		#cuda
		input_data = input_data.to(device, dtype=torch.float)
		gt_data = gt_data.to(device, dtype=torch.float)
		mask_data = mask_data.to(device, dtype=torch.float)
		#dk_data = dk_data.to(device, dtype=torch.float)

		optimizer.zero_grad()

		output_data = model(input_data, dk_data, mask_data.unsqueeze(1))
		#output_data = model(input_data)	

		loss = loss_fn(output_data, gt_data)

		loss.backward()

		gradient_clipping_list = ['vdsrr']

		if args.model_arch in gradient_clipping_list:
			nn.utils.clip_grad_norm_(model.parameters(), 0.4)

		optimizer.step()

		per_loss = loss.item()

		tb_writer.add_scalar('train/overall_loss', per_loss, total_tb_it)
		total_tb_it += 1

		if batch_count%print_freq == 0:
			print('Epoch [%d/%d] Loss: %.8f' %(epoch, args.num_epoch, per_loss))

	return total_tb_it

def validate(device, model, val_loader, epoch, loss_fn, tb_writer):

	model.eval()

	tb_loss = 0
	mse_loss = 0
	ssim_perf = 0
	psnr_perf = 0

	with torch.no_grad():

		for batch_count, (input_data, gt_data, mask_data, dk_data, input_name) in enumerate(val_loader):

			#cuda
			input_data = input_data.to(device, dtype=torch.float)
			gt_data = gt_data.to(device, dtype=torch.float)
			mask_data = mask_data.to(device, dtype=torch.float)
			#dk_data = dk_data.to(device, dtype=torch.float)
	
			output_data = model(input_data, dk_data, mask_data.unsqueeze(1))
			#output_data = model(input_data)			

			loss = loss_fn(output_data, gt_data)

			tb_loss += loss.item()

			mask = torch.squeeze(mask_data, 0).cpu().numpy()[:,:,:,np.newaxis]
			
			if data_normalization:
				og_output = torch.squeeze(output_data, 0).permute(1, 2, 3, 0).cpu().numpy() * val_loader.dataset.gt_std
				og_output = og_output + val_loader.dataset.gt_mean
				og_output = og_output * mask

				og_gt = torch.squeeze(gt_data, 0).permute(1, 2, 3, 0).cpu().numpy() * val_loader.dataset.gt_std
				og_gt = og_gt + val_loader.dataset.gt_mean
				og_gt = og_gt * mask

			else:
				og_output = torch.squeeze(output_data, 0).permute(1, 2, 3, 0).cpu().numpy() * mask
				og_gt = torch.squeeze(gt_data, 0).permute(1, 2, 3, 0).cpu().numpy() * mask

			mse_loss += qsm_mse(og_gt, og_output, mask, roi=True)
			psnr_perf += qsm_psnr(og_gt, og_output, mask, root_dir, roi=True)
			ssim_perf += qsm_ssim(og_gt, og_output, mask, root_dir)

		avg_tb_loss = tb_loss / len(val_loader.dataset)
		avg_mse_loss = mse_loss / len(val_loader.dataset) / validate_mse
		avg_psnr_perf = psnr_perf / len(val_loader.dataset)
		avg_ssim_perf = ssim_perf / len(val_loader.dataset)
		print('alpha: %.3f' %(model.alpha.cpu().numpy()))
		print('##Validate loss: %.8f Mse: %.8f PSNR: %.8f SSIM: %.8f' %(avg_tb_loss, avg_mse_loss, avg_psnr_perf, avg_ssim_perf))

		tb_writer.add_scalar('val/overall_loss', avg_tb_loss, epoch)
		tb_writer.add_scalar('val/Mse', avg_mse_loss, epoch)
		tb_writer.add_scalar('val/PSNR', avg_psnr_perf, epoch)
		tb_writer.add_scalar('val/SSIM', avg_ssim_perf, epoch)

	return avg_mse_loss

def predict(args, device, model, data_loader, loss_fn, model_name):

	nifti_path = root_dir + 'qsm_dataset/qsm_data/'

	model.eval()

	mse_loss = 0
	ssim_perf = 0
	w_psnr_perf = 0
	r_psnr_perf = 0

	if case == 'patch':
		with torch.no_grad():
			for batch_count, (input_data, gt_data, mask_data, dk_data, input_name) in enumerate(data_loader):
				
				subject = input_name[0][:6]
				ori = input_name[0][7:11]					

				#cuda
				input_data = input_data.to(device, dtype=torch.float)
				gt_data = gt_data.to(device, dtype=torch.float)
				mask_data = mask_data.to(device, dtype=torch.float)
				#dk_data = dk_data.to(device, dtype=torch.float)

				output_data = model(input_data, dk_data, mask_data.unsqueeze(1))
				
				mask = torch.squeeze(mask_data, 0).cpu().numpy()[:,:,:,np.newaxis]

				if data_normalization:
					og_output = torch.squeeze(output_data, 0).permute(1, 2, 3, 0).cpu().numpy() * data_loader.dataset.gt_std
					og_output = og_output + data_loader.dataset.gt_mean
					og_output = og_output * mask
					#og_output = torch.squeeze(output_data, 0).permute(1, 2, 3, 0).cpu().numpy() * mask

					og_gt = torch.squeeze(gt_data, 0).permute(1, 2, 3, 0).cpu().numpy() * data_loader.dataset.gt_std
					og_gt = og_gt + data_loader.dataset.gt_mean
					og_gt = og_gt * mask
					#og_gt = torch.squeeze(gt_data, 0).permute(1, 2, 3, 0).cpu().numpy() * mask

				else:
					og_output = torch.squeeze(output_data, 0).permute(1, 2, 3, 0).cpu().numpy() * mask
					og_gt = torch.squeeze(gt_data, 0).permute(1, 2, 3, 0).cpu().numpy() * mask

				mse_loss += qsm_mse(og_gt, og_output, mask, roi=True)
				ssim_perf += qsm_ssim(og_gt, og_output, mask, root_dir)	
				w_psnr_perf += qsm_psnr(og_gt, og_output, mask, root_dir, roi=False)
				r_psnr_perf += qsm_psnr(og_gt, og_output, mask, root_dir, roi=True)

				if not args.no_save:
				
					qsm_path = nifti_path + subject + '/cosmos/' + subject + '_cosmos.nii.gz'

					if not os.path.exists(vis_output_path + model_name):
						os.makedirs(vis_output_path + model_name)

					save_name = vis_output_path + model_name + '/' + input_name[0][:-4] + '_pred'

					qsm_display(og_output, qsm_path, torch.squeeze(mask_data, 0).cpu().numpy(), out_name=save_name)

			avg_mse_loss = mse_loss / len(data_loader.dataset) / validate_mse
			avg_ssim_perf = ssim_perf / len(data_loader.dataset)
			avg_w_psnr_perf = w_psnr_perf / len(data_loader.dataset)
			avg_r_psnr_perf = r_psnr_perf / len(data_loader.dataset)

			print('##Test Mse: %.8f PSNR: %.8f(%.8f) SSIM: %.8f' %(avg_mse_loss, avg_w_psnr_perf, avg_r_psnr_perf, avg_ssim_perf))
				
	elif case == 'whole':
		with torch.no_grad():
			for batch_count, (input_data, gt_data, mask_data, dk_data, input_name) in enumerate(data_loader):

				subject = input_name[0][:6]
				ori = input_name[0][7:11] 
				iter_num = 1
				#cuda
				input_data = input_data.to(device, dtype=torch.float)
				gt_data = gt_data.to(device, dtype=torch.float)
				mask_data = mask_data.to(device, dtype=torch.float)
				#dk_data = dk_data.to(device, dtype=torch.float)

				_, _, x_len, y_len, z_len = input_data.shape
				pad_list = ['pudnn']

				if args.model_arch in pad_list:
					#pad_x = x_len - ((x_len//16))*16
					#pad_y = y_len - ((y_len//16))*16
					pad_z = ((z_len//16)+1)*16 - z_len
					pad_fn = nn.ConstantPad3d((0, pad_z, 0, 0, 0, 0), 0)
					input_data = pad_fn(input_data)
					#input_data = input_data[:, :, :, :, int(pad_z/2):int(-pad_z/2)]
					
				output_data = model(input_data, dk_data, mask_data.unsqueeze(1))	
				#output_data = model(input_data, dk_data, pad_fn(mask_data).unsqueeze(1), False)#[:, :, :, :, int(pad_z/2):int(-pad_z/2)])
				#output_data = model(input_data)

				if args.model_arch in pad_list:
					#pad_x = ((x_len//16)+1)*16 - x_len
					#pad_y = ((y_len//16)+1)*16 - y_len
					pad_z = ((z_len//16)+1)*16 - z_len
					#pad_x = x_len - input_data.shape[2]
					#pad_y = y_len - input_data.shape[3]
					#pad_z = z_len - input_data.shape[4]
					#pad_fn = nn.ConstantPad3d((int(pad_z/2), int(pad_z/2), int(pad_y/2), int(pad_y/2), int(pad_x/2), int(pad_x/2)), 0)
					#output_data[-1] = pad_fn(output_data[-1])
					output_data = output_data[:, :, :, :, :-pad_z]

				mask = torch.squeeze(mask_data, 0).cpu().numpy()[:,:,:,np.newaxis]

				if data_normalization:
					og_output = torch.squeeze(output_data, 0).permute(1, 2, 3, 0).cpu().numpy() * data_loader.dataset.gt_std
					og_output = og_output + data_loader.dataset.gt_mean
					og_output = og_output * mask
					#og_output = torch.squeeze(output_data, 0).permute(1, 2, 3, 0).cpu().numpy() * mask	

					og_gt = torch.squeeze(gt_data, 0).permute(1, 2, 3, 0).cpu().numpy() * data_loader.dataset.gt_std
					og_gt = og_gt + data_loader.dataset.gt_mean
					og_gt = og_gt * mask
					#og_gt = torch.squeeze(gt_data, 0).permute(1, 2, 3, 0).cpu().numpy() * mask

				else:
					og_output = torch.squeeze(output_data, 0).permute(1, 2, 3, 0).cpu().numpy() * mask
					og_gt = torch.squeeze(gt_data, 0).permute(1, 2, 3, 0).cpu().numpy() * mask

				mse_loss += qsm_mse(og_gt, og_output, mask, roi=True)
				ssim_perf += qsm_ssim(og_gt, og_output, mask, root_dir)	
				w_psnr_perf += qsm_psnr(og_gt, og_output, mask, root_dir, roi=False)
				r_psnr_perf += qsm_psnr(og_gt, og_output, mask, root_dir, roi=True)
				print('##Test PSNR: %.8f SSIM: %.8f' %(qsm_psnr(og_gt, og_output, mask, root_dir, roi=True), qsm_ssim(og_gt, og_output, mask, root_dir)))

				if not args.no_save:

					qsm_path = nifti_path + subject + '/cosmos/' + subject + '_cosmos.nii.gz'

					if not os.path.exists(vis_output_path + model_name):
						os.makedirs(vis_output_path + model_name)

					save_name = vis_output_path + model_name + '/' + input_name[0][:-4] + '_pred' + str(iter_num)

					qsm_display(og_output, qsm_path, torch.squeeze(mask_data, 0).cpu().numpy(), out_name=save_name)
					
			avg_mse_loss = mse_loss / len(data_loader.dataset) / whole_validate_mse
			avg_ssim_perf = ssim_perf / len(data_loader.dataset)
			avg_w_psnr_perf = w_psnr_perf / len(data_loader.dataset)
			avg_r_psnr_perf = r_psnr_perf / len(data_loader.dataset)

			print('##Test Mse: %.8f PSNR: %.8f(%.8f) SSIM: %.8f' %(avg_mse_loss, avg_w_psnr_perf, avg_r_psnr_perf, avg_ssim_perf))


parser = argparse.ArgumentParser(description='QSM Inversion problem')
parser.add_argument('--mode', default='train', choices=['train', 'predict'], help='operation mode: train or predict (default: train)')
parser.add_argument('--name', type=str, default='_test', help='the name of exp')
parser.add_argument('--dataset', default='qsm', choices=['qsm'], help='dataset to use (default: qsm)')
parser.add_argument('--tesla', default=7, type=int, choices=[3, 7], help='B0 tesla(default: 7)')
parser.add_argument('--gpu_num', default=1, type=int, choices=[1, 2, 3, 4], help='number of gpu (default: 1)')
parser.add_argument('--model_arch', default='vdsr', choices=['vdsr', 'vdsrr', 'pdnn', 'pldnn', 'pndnn', 'wpdnn', 'wpldnn', 'pudnn', 'pesp'], help='network model (default: vdsrr)')
parser.add_argument('--num_epoch', default=600, type=int, metavar='N', help='number of total epochs to run (default: 20)')
parser.add_argument('--batch_size', type=int, default=10, help='batch size (default: 4)')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam', 'sgdadam'], help='optimizer to use (default: adam)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--no_save', action='store_true', default=False, help='disables saving tensors')
parser.add_argument('--resume_file', type=str, default=None, help='the checkpoint file to resume from')

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)
