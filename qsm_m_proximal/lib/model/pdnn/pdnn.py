import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Conv_ReLU_Block(nn.Module):
	def __init__(self):
		super(Conv_ReLU_Block, self).__init__()
		self.conv = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
		self.relu = nn.ReLU(inplace=True)
		
	def forward(self, x):
		return self.relu(self.conv(x))

class BasicBlock(nn.Module):

    def __init__(self, inplanes=32, planes=32):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

		
class PDNN(nn.Module):
	def __init__(self, gt_mean, gt_std):
		super().__init__()
		self.gt_mean = torch.from_numpy(np.load(gt_mean)).float()
		self.gt_std = torch.from_numpy(np.load(gt_std)).float()

		self.iter_num = 3

		self.alpha = torch.nn.Parameter(torch.ones(1)*4)
		#self.alpha = torch.nn.Parameter(torch.ones(self.iter_num))

		self.gen = nn.Sequential(
				nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
				nn.ReLU(inplace=True),
				self.make_layer(BasicBlock, 8),
				nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
		)
				
	def make_layer(self, block, num_of_layer):
		layers = []
		for _ in range(num_of_layer):
			layers.append(block())
		return nn.Sequential(*layers)

	def forward(self, y, dk, mask):

		batch_size, _, x_dim, y_dim, z_dim = y.shape

		out = []

		dk_batch = []
		dim1_batch = []
		dim2_batch = []
		dim3_batch = []
		
		x_est = torch.empty_like(y)
		
		for b_n in range(batch_size):
			dk_batch.append(torch.from_numpy(np.load(dk[b_n])[np.newaxis, :, :, :, np.newaxis]).to(y.device, dtype=torch.float))

			dim1_batch.append(dk_batch[-1].shape[1])
			dim2_batch.append(dk_batch[-1].shape[2])
			dim3_batch.append(dk_batch[-1].shape[3])

			x_est[b_n, :, :, :, :] = self.alpha * torch.irfft(dk_batch[-1] * torch.rfft(F.pad(y[b_n, :, :, :, :], (0, dim3_batch[-1]-z_dim, 0, dim2_batch[-1]-y_dim, 0, dim1_batch[-1]-x_dim)), 3, normalized=True, onesided=False), 3, normalized=True, onesided=False)[:, :x_dim, :y_dim, :z_dim]


		for i in range(self.iter_num):

			if i == 0:
				pn_x_pred = 0
			else:
				pn_x_pred = torch.empty_like(y)

				for b_n in range(batch_size):
					pn_x_pred[b_n, :, :, :, :] = den_x_pred[b_n, :, :, :, :] - self.alpha * torch.irfft(dk_batch[b_n] * dk_batch[b_n] * torch.rfft(F.pad(den_x_pred[b_n, :, :, :, :], (0, dim3_batch[b_n]-z_dim, 0, dim2_batch[b_n]-y_dim, 0, dim1_batch[b_n]-x_dim)), 3, normalized=True, onesided=False), 3, normalized=True, onesided=False)[:, :x_dim, :y_dim, :z_dim]

			x_input = (((x_est + pn_x_pred) - self.gt_mean) / self.gt_std) * mask
			x_pred = self.gen(x_input)
			den_x_pred = ((x_pred * self.gt_std) + self.gt_mean) * mask

			out.append(x_pred)
	
		return out
