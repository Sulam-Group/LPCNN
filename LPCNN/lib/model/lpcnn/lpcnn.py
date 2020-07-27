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

class wBasicBlock(nn.Module):

	def __init__(self, inplanes=32, planes=32, dropout_rate=0.5):
		super(wBasicBlock, self).__init__()
		self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
		self.bn1 = nn.BatchNorm3d(planes)
		self.relu = nn.ReLU(inplace=True)
	
		self.dropout = nn.Dropout3d(p=dropout_rate)
	
		self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
		self.bn2 = nn.BatchNorm3d(planes)

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.dropout(out)

		out = self.conv2(out)
		out = self.bn2(out)

		out += residual
		out = self.relu(out)

		return out

		
class LPCNN(nn.Module):
	def __init__(self, gt_mean, gt_std):
		super().__init__()
		self.gt_mean = torch.from_numpy(np.load(gt_mean)).float()
		self.gt_std = torch.from_numpy(np.load(gt_std)).float()

		self.iter_num = 3

		self.alpha = torch.nn.Parameter(torch.ones(1)*4)
		#self.alpha = torch.nn.Parameter(torch.ones(self.iter_num)*4)

		self.gen = nn.Sequential(
				nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
				nn.ReLU(inplace=True),
				self.make_layer(wBasicBlock, 8),
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

		batch_size, _, x_dim, y_dim, z_dim, number = y.shape

		out = []
		
		dk_batch = []
		dim1_batch = []
		dim2_batch = []
		dim3_batch = []
		
		x_est = []

		for num in range(number):
			x_est.append(torch.empty_like(y[:, :, :, :, :, num]))

		for b_n in range(batch_size):
			dk_batch.append([])
			dim1_batch.append([])
			dim2_batch.append([])
			dim3_batch.append([])

			dk_list = dk[b_n].split(' ')[:-1]
			for num in range(number):
				dk_batch[-1].append(torch.from_numpy(np.load(dk_list[num])[np.newaxis, :, :, :, np.newaxis]).to(y.device, dtype=torch.float))

				dim1_batch[-1].append(dk_batch[-1][-1].shape[1])
				dim2_batch[-1].append(dk_batch[-1][-1].shape[2])
				dim3_batch[-1].append(dk_batch[-1][-1].shape[3])			


				x_est[num][b_n, :, :, :, :] = self.alpha * torch.ifft(dk_batch[-1][num] * torch.rfft(F.pad(y[b_n, :, :, :, :, num], (0, dim3_batch[-1][num]-z_dim, 0, dim2_batch[-1][num]-y_dim, 0, dim1_batch[-1][num]-x_dim)), 3, normalized=True, onesided=False), 3, normalized=True)[:, :x_dim, :y_dim, :z_dim, 0]


		for i in range(self.iter_num):

			if i == 0:
				pn_x_pred = torch.zeros_like(y[:, :, :, :, :, num])

				for num in range(number):
					pn_x_pred += x_est[num]
			else:
				pn_x_pred = den_x_pred
		
				for b_n in range(batch_size):
					for num in range(number):
						pn_x_pred[b_n, :, :, :, :] += x_est[num][b_n, :, :, :, :] - self.alpha * torch.ifft(dk_batch[b_n][num] * dk_batch[b_n][num] * torch.rfft(F.pad(den_x_pred[b_n, :, :, :, :], (0, dim3_batch[b_n][num]-z_dim, 0, dim2_batch[b_n][num]-y_dim, 0, dim1_batch[b_n][num]-x_dim)), 3, normalized=True, onesided=False), 3, normalized=True)[:, :x_dim, :y_dim, :z_dim, 0]

			x_input = ((pn_x_pred - self.gt_mean) / self.gt_std) * mask
			x_pred = self.gen(x_input)
			den_x_pred = ((x_pred * self.gt_std) + self.gt_mean) * mask

		return x_pred

