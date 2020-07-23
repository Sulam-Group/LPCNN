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

from lib.model.gan_wpdnn.gan_wpdnn import *
from lib.loss.fwl1.fwl1 import fwl1

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
		g_model = WPDNN(train_loader.dataset.root + 'train_gt_mean.npy', train_loader.dataset.root + 'train_gt_std.npy')
		g_model.to(device)

		d_model = Discriminator()
		d_model.to(device)
		print(args.model_arch + ' loaded.')

		## parallel model
		if args.gpu_num > 1:
			g_model = nn.DataParallel(g_model)
			d_model = nn.DataParallel(d_model)

		## loss function and optimizer
		g_loss_fn = fwl1
		d_loss_fn = nn.MSELoss()
		g_optimizer = optim.Adam(g_model.parameters(), args.learning_rate, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
		d_optimizer = optim.Adam(d_model.parameters(), args.learning_rate, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

		#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
		g_scheduler = optim.lr_scheduler.ExponentialLR(g_optimizer, gamma=0.98)
		d_scheduler = optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=0.98)

		## initailize statistic result
		start_epoch = 0
		best_per_index = 1000
		total_tb_it = 0

		## resume training
		if not args.resume_file == None:
			start_epoch = torch.load(args.resume_file, map_location=device)['epoch'] + 1
			g_model.load_state_dict(torch.load(args.resume_file, map_location=device)['g_model_state'])
			g_model.to(device)

			d_model.load_state_dict(torch.load(args.resume_file, map_location=device)['d_model_state'])
			d_model.to(device)
			print(args.resume_file + ' loaded.')
		
			g_optimizer.load_state_dict(torch.load(args.resume_file, map_location=device)['g_optimizer_state'])
			d_optimizer.load_state_dict(torch.load(args.resume_file, map_location=device)['d_optimizer_state'])

		for epoch in range(start_epoch, args.num_epoch):
			
			total_tb_it = train(args, device, g_model, d_model, train_loader, epoch, g_loss_fn, d_loss_fn, g_optimizer, d_optimizer, tb_writer, total_tb_it)
			mse_index = validate(device, g_model, d_model, val_loader, epoch, g_loss_fn, d_loss_fn, tb_writer)

			for param_group in g_optimizer.param_groups:
				print('Learning rate: %.8f' %(param_group['lr']))	

			g_scheduler.step(epoch)
			d_scheduler.step(epoch)

			state = {'epoch': epoch, 'g_model_state': g_model.state_dict(), 'd_model_state': d_model.state_dict(), 'g_optimizer_state': g_optimizer.state_dict(), 'd_optimizer_state': d_optimizer.state_dict()}

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
		model = WPDNN(data_loader.dataset.root + 'train_gt_mean.npy', data_loader.dataset.root + 'train_gt_std.npy')
		model.load_state_dict(torch.load(args.resume_file, map_location=device)['g_model_state'])
		model.to(device)
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

def train(args, device, g_model, d_model, train_loader, epoch, g_loss_fn, d_loss_fn, g_optimizer, d_optimizer, tb_writer, total_tb_it):

	print_freq = (len(train_loader.dataset) // args.batch_size) // 3

	g_model.train()
	d_model.train()

	for batch_count, (input_data, gt_data, mask_data, dk_data, input_name) in enumerate(train_loader):

		#cuda
		input_data = input_data.to(device, dtype=torch.float)
		gt_data = gt_data.to(device, dtype=torch.float)
		mask_data = mask_data.to(device, dtype=torch.float)
		#dk_data = dk_data.to(device, dtype=torch.float)

		#Discriminator
		d_optimizer.zero_grad()

		input_r = gt_data
		input_f = g_model(input_data, dk_data, mask_data.unsqueeze(1))[-1].detach()

		##Soft target
		target_r = (torch.rand(args.batch_size, 1)*0.1 + 0.9).to(device, dtype=torch.float)
		target_f = (torch.rand(args.batch_size, 1)*0.1).to(device, dtype=torch.float)

		d_output_r = d_model(input_r)
		d_output_f = d_model(input_f)

		d_loss_r = d_loss_fn(d_output_r, target_r)
		d_loss_f = d_loss_fn(d_output_f, target_f)

		d_loss = d_loss_r + d_loss_f
		
		d_loss.backward()

		d_optimizer.step()

		#Generator
		g_optimizer.zero_grad()

		output_data = g_model(input_data, dk_data, mask_data.unsqueeze(1))

		d_output = d_model(output_data[-1])

		g_loss = g_loss_fn(output_data, gt_data, mask_data.unsqueeze(1), input_data, dk_data, train_loader.dataset.root)

		a_loss = d_loss_fn(d_output, torch.ones([args.batch_size, 1]).to(device, dtype=torch.float))

		total_g_loss = g_loss + 1e-2 * a_loss

		total_g_loss.backward()

		g_optimizer.step()

		gradient_clipping_list = ['vdsrr']

		if args.model_arch in gradient_clipping_list:
			nn.utils.clip_grad_norm_(model.parameters(), 0.4)

		per_d_loss = d_loss.item()
		per_tg_loss = total_g_loss.item()
		per_g_loss = g_loss.item()
		per_a_loss = a_loss.item()

		tb_writer.add_scalar('train/d_loss', per_d_loss, total_tb_it)
		tb_writer.add_scalar('train/total_g_loss', per_tg_loss, total_tb_it)
		tb_writer.add_scalar('train/g_loss', per_g_loss, total_tb_it)
		tb_writer.add_scalar('train/a_loss', per_a_loss, total_tb_it)
		total_tb_it += 1

		if batch_count%print_freq == 0:
			print('Epoch [%d/%d] D_Loss: %.8f\tG_Loss: %.8f\tA_Loss: %.8f\ttotal_G_Loss: %.8f' %(epoch, args.num_epoch, per_d_loss, per_g_loss, per_a_loss, per_tg_loss))
	return total_tb_it

def validate(device, g_model, d_model, val_loader, epoch, g_loss_fn, d_loss_fn, tb_writer):

	d_model.eval()
	g_model.eval()

	tb_d_loss = 0
	tb_tg_loss = 0
	tb_g_loss = 0
	tb_a_loss = 0

	mse_loss = 0
	ssim_perf = 0
	psnr_perf = 0

	real_mean = 0
	fake_mean = 0

	with torch.no_grad():

		for batch_count, (input_data, gt_data, mask_data, dk_data, input_name) in enumerate(val_loader):

			#cuda
			input_data = input_data.to(device, dtype=torch.float)
			gt_data = gt_data.to(device, dtype=torch.float)
			mask_data = mask_data.to(device, dtype=torch.float)
			#dk_data = dk_data.to(device, dtype=torch.float)

			output_data = g_model(input_data, dk_data, mask_data.unsqueeze(1))

			d_output_r = d_model(gt_data)
			d_output_f = d_model(output_data[-1])

			d_loss = d_loss_fn(d_output_r, torch.ones([1, 1]).to(device, dtype=torch.float)) + d_loss_fn(d_output_f, torch.zeros([1, 1]).to(device, dtype=torch.float))

			g_loss = g_loss_fn(output_data, gt_data, mask_data.unsqueeze(1), input_data, dk_data, val_loader.dataset.root)

			a_loss = d_loss_fn(d_output_f, torch.ones([1, 1]).to(device, dtype=torch.float))

			total_g_loss = g_loss + 1e-2 * a_loss

			tb_d_loss += d_loss.item()
			tb_g_loss += g_loss.item()
			tb_a_loss += a_loss.item()
			tb_tg_loss += total_g_loss.item()

			real_mean += d_output_r
			fake_mean += d_output_f

			mask = torch.squeeze(mask_data, 0).cpu().numpy()[:,:,:,np.newaxis]
			
			if data_normalization:
				og_output = torch.squeeze(output_data[-1], 0).permute(1, 2, 3, 0).cpu().numpy() * val_loader.dataset.gt_std
				og_output = og_output + val_loader.dataset.gt_mean
				og_output = og_output * mask
				#og_output = torch.squeeze(output_data, 0).permute(1, 2, 3, 0).cpu().numpy() * mask

				og_gt = torch.squeeze(gt_data, 0).permute(1, 2, 3, 0).cpu().numpy() * val_loader.dataset.gt_std
				og_gt = og_gt + val_loader.dataset.gt_mean
				og_gt = og_gt * mask
				#og_gt = torch.squeeze(gt_data, 0).permute(1, 2, 3, 0).cpu().numpy() * mask

			else:
				og_output = torch.squeeze(output_data[-1], 0).permute(1, 2, 3, 0).cpu().numpy() * mask
				og_gt = torch.squeeze(gt_data, 0).permute(1, 2, 3, 0).cpu().numpy() * mask

			mse_loss += qsm_mse(og_gt, og_output, mask, roi=True)
			psnr_perf += qsm_psnr(og_gt, og_output, mask, root_dir, roi=True)
			ssim_perf += qsm_ssim(og_gt, og_output, mask, root_dir)

		avg_tb_d_loss = tb_d_loss / len(val_loader.dataset)
		avg_tb_g_loss = tb_g_loss / len(val_loader.dataset)
		avg_tb_a_loss = tb_a_loss / len(val_loader.dataset)
		avg_tb_tg_loss = tb_tg_loss / len(val_loader.dataset)

		real_mean = real_mean / len(val_loader.dataset)
		fake_mean = fake_mean / len(val_loader.dataset)

		avg_mse_loss = mse_loss / len(val_loader.dataset) / validate_mse
		avg_psnr_perf = psnr_perf / len(val_loader.dataset)
		avg_ssim_perf = ssim_perf / len(val_loader.dataset)
		#print('alpha: %.3f' %(model.alpha.numpy()))
		print('##Validata D_Loss: %.8f Total_G_Loss: %.8f' %(avg_tb_d_loss, avg_tb_tg_loss))
		print('##Validate real_mean: %.4f fake_mean: %.4f Mse: %.8f PSNR: %.8f SSIM: %.8f' %(real_mean, fake_mean, avg_mse_loss, avg_psnr_perf, avg_ssim_perf))

		tb_writer.add_scalar('val/d_loss', avg_tb_d_loss, epoch)
		tb_writer.add_scalar('val/g_loss', avg_tb_g_loss, epoch)
		tb_writer.add_scalar('val/a_loss', avg_tb_a_loss, epoch)
		tb_writer.add_scalar('val/total_g_loss', avg_tb_tg_loss, epoch)

		tb_writer.add_scalar('val/real_mean', real_mean, epoch)
		tb_writer.add_scalar('val/fake_mean', fake_mean, epoch)

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
					og_output = torch.squeeze(output_data[-1], 0).permute(1, 2, 3, 0).cpu().numpy() * data_loader.dataset.gt_std
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

				output_data = model(input_data, dk_data, mask_data.unsqueeze(1))
				#output_data = model(input_data)

				mask = torch.squeeze(mask_data, 0).cpu().numpy()[:,:,:,np.newaxis]

				if data_normalization:
					og_output = torch.squeeze(output_data[-1* iter_num], 0).permute(1, 2, 3, 0).cpu().numpy() * data_loader.dataset.gt_std
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
parser.add_argument('--model_arch', default='vdsr', choices=['vdsr', 'vdsrr', 'pdnn', 'pldnn', 'pndnn', 'wpdnn', 'wpldnn'], help='network model (default: vdsrr)')
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
