
import numpy as np

import torch
import torch.nn.functional as F

def fwl1(output, groundtruth, mask, phase, dipole, root_dir):

	gt_mean = torch.from_numpy(np.load(root_dir + 'train_gt_mean.npy')).float()	
	gt_std = torch.from_numpy(np.load(root_dir + 'train_gt_std.npy')).float()

	input_mean = torch.from_numpy(np.load(root_dir + 'train_input_mean.npy')).float()
	input_std = torch.from_numpy(np.load(root_dir + 'train_input_std.npy')).float()

	batch_size, _, x_dim, y_dim, z_dim, number = phase.shape

	dk_batch = []
	dim1_batch = []
	dim2_batch = []
	dim3_batch = []

	for b_n in range(batch_size):
		dk_batch.append([])
		dim1_batch.append([])
		dim2_batch.append([])
		dim3_batch.append([])

		dk_list = dipole[b_n].split(' ')[:-1]
		for num in range(number):
			dk_batch[-1].append(torch.from_numpy(np.load(dk_list[num])[np.newaxis, :, :, :, np.newaxis]).to(phase.device, dtype=torch.float))
			dim1_batch[-1].append(dk_batch[-1][-1].shape[1])
			dim2_batch[-1].append(dk_batch[-1][-1].shape[2])
			dim3_batch[-1].append(dk_batch[-1][-1].shape[3])

	loss = 0

	for i in range(len(output)):
		
		for num in range(number):

			recon_phase = torch.empty_like(phase[:, :, :, :, :, num])

			for b_n in range(batch_size):
				recon_phase[b_n, :, :, :, :] = torch.irfft(dk_batch[b_n][num] * torch.rfft(F.pad(((output[i][b_n, :, :, :, :]*gt_std) + gt_mean) * mask[b_n, :, :, :, :, num], (0, dim3_batch[b_n][num]-z_dim, 0, dim2_batch[b_n][num]-y_dim, 0, dim1_batch[b_n][num]-x_dim)), 3, normalized=True, onesided=False), 3, normalized=True, onesided=False)[:, :x_dim, :y_dim, :z_dim]
		
			norm_recon = ((recon_phase - input_mean) / input_std) * mask[:, :, :, :, :, 0]

			norm_phase = ((phase[:, :, :, :, :, num] - input_mean) / input_std) * mask[:, :, :, :, :, 0]

			loss += F.mse_loss(norm_recon[:, :, 5:-5, 5:-5, 5:-5], norm_phase[:, :, 5:-5, 5:-5, 5:-5])
			#loss += F.l1_loss(output[i], groundtruth)	

	loss = loss / len(output)
	#print(loss)
	f_loss = F.mse_loss(output[-1], groundtruth)
	#print(f_loss)
	loss = loss * 0.2 + f_loss * 0.8

	return loss
