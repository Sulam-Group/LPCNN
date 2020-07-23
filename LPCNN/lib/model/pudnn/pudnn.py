import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class UNET(nn.Module):
	def __init__(self, in_channel=1, out_channel=1):
		super(UNET, self).__init__()

		self.in_channel = in_channel
		self.out_channel = out_channel

		self.conv11 = nn.Sequential(nn.Conv3d(self.in_channel, 32, kernel_size=5, padding=2), nn.BatchNorm3d(32), nn.ReLU(inplace=True))
		self.conv12 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=5, padding=2), nn.BatchNorm3d(32), nn.ReLU(inplace=True))

		self.maxpool2m = nn.MaxPool3d(2)
		self.conv21 = nn.Sequential(nn.Conv3d(32, 64, kernel_size=5, padding=2), nn.BatchNorm3d(64), nn.ReLU(inplace=True))
		self.conv22 = nn.Sequential(nn.Conv3d(64, 64, kernel_size=5, padding=2), nn.BatchNorm3d(64), nn.ReLU(inplace=True))

		self.maxpool3m = nn.MaxPool3d(2)
		self.conv31 = nn.Sequential(nn.Conv3d(64, 128, kernel_size=5, padding=2), nn.BatchNorm3d(128), nn.ReLU(inplace=True))
		self.conv32 = nn.Sequential(nn.Conv3d(128, 128, kernel_size=5, padding=2), nn.BatchNorm3d(128), nn.ReLU(inplace=True))

		self.maxpool4m = nn.MaxPool3d(2)
		self.conv41 = nn.Sequential(nn.Conv3d(128, 256, kernel_size=5, padding=2), nn.BatchNorm3d(256), nn.ReLU(inplace=True))
		self.conv42 = nn.Sequential(nn.Conv3d(256, 256, kernel_size=5, padding=2), nn.BatchNorm3d(256), nn.ReLU(inplace=True))

		self.maxpool5m = nn.MaxPool3d(2)
		self.conv51 = nn.Sequential(nn.Conv3d(256, 512, kernel_size=5, padding=2), nn.BatchNorm3d(512), nn.ReLU(inplace=True))
		self.conv52 = nn.Sequential(nn.Conv3d(512, 512, kernel_size=5, padding=2), nn.BatchNorm3d(512), nn.ReLU(inplace=True))

		self.deconv61 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2, padding=0)
		self.conv62 = nn.Sequential(nn.Conv3d(512, 256, kernel_size=5, padding=2), nn.BatchNorm3d(256), nn.ReLU(inplace=True))
		self.conv63 = nn.Sequential(nn.Conv3d(256, 256, kernel_size=5, padding=2), nn.BatchNorm3d(256), nn.ReLU(inplace=True))

		self.deconv71 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2, padding=0)
		self.conv72 = nn.Sequential(nn.Conv3d(256, 128, kernel_size=5, padding=2), nn.BatchNorm3d(128), nn.ReLU(inplace=True))
		self.conv73 = nn.Sequential(nn.Conv3d(128, 128, kernel_size=5, padding=2), nn.BatchNorm3d(128), nn.ReLU(inplace=True))

		self.deconv81 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, padding=0)
		self.conv82 = nn.Sequential(nn.Conv3d(128, 64, kernel_size=5, padding=2), nn.BatchNorm3d(64), nn.ReLU(inplace=True))
		self.conv83 = nn.Sequential(nn.Conv3d(64, 64, kernel_size=5, padding=2), nn.BatchNorm3d(64), nn.ReLU(inplace=True))

		self.deconv91 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, padding=0)
		self.conv92 = nn.Sequential(nn.Conv3d(64, 32, kernel_size=5, padding=2), nn.BatchNorm3d(32), nn.ReLU(inplace=True))
		self.conv93 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=5, padding=2), nn.BatchNorm3d(32), nn.ReLU(inplace=True))

		self.conv101 = nn.Conv3d(32, 1, kernel_size=1, padding=0)

		self.dropout1 = nn.Dropout3d(p=0.5)
		self.dropout2 = nn.Dropout3d(p=0.5)
		self.dropout3 = nn.Dropout3d(p=0.5)
		self.dropout4 = nn.Dropout3d(p=0.5)

		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				nn.init.xavier_uniform_(m.weight)

	def forward(self, x):

		x1 = self.conv12(self.conv11(x))
		x2 = self.conv22(self.conv21(self.maxpool2m(x1)))
		x3 = self.conv32(self.conv31(self.maxpool3m(x2)))
		x4 = self.conv42(self.conv41(self.maxpool4m(x3)))
		x5 = self.conv52(self.conv51(self.maxpool5m(x4)))
	
		x = self.conv63(self.conv62(self.dropout1(torch.cat((self.deconv61(x5), x4), dim=1))))	
		x = self.conv73(self.conv72(self.dropout2(torch.cat((self.deconv71(x), x3), dim=1))))
		x = self.conv83(self.conv82(self.dropout3(torch.cat((self.deconv81(x), x2), dim=1))))
		x = self.conv93(self.conv92(self.dropout4(torch.cat((self.deconv91(x), x1), dim=1))))

		out = self.conv101(x)

		return out
		
class PUDNN(nn.Module):
	def __init__(self, gt_mean, gt_std):
		super().__init__()
		self.gt_mean = torch.from_numpy(np.load(gt_mean)).float()
		self.gt_std = torch.from_numpy(np.load(gt_std)).float()

		self.iter_num = 3

		self.alpha = torch.nn.Parameter(torch.ones(1)*4)
		#self.alpha = torch.nn.Parameter(torch.ones(self.iter_num))

		self.gen = UNET()

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
