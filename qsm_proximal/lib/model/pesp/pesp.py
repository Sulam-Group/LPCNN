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

class BR(nn.Module):
	'''
		This class groups the batch normalization and PReLU activation
	'''
	def __init__(self, nOut):
		'''
		:param nOut: output feature maps
		'''
		super().__init__()
		self.bn = nn.BatchNorm3d(nOut, eps=1e-03)
		self.act = nn.PReLU(nOut)

	def forward(self, input):
		'''
		:param input: input feature map
		:return: normalized and thresholded feature map
		'''
		output = self.bn(input)
		output = self.act(output)
		return output

class C(nn.Module):
	'''
	This class is for a convolutional layer.
	'''
	def __init__(self, nIn, nOut, kSize, stride=1):
		'''

		:param nIn: number of input channels
		:param nOut: number of output channels
		:param kSize: kernel size
		:param stride: optional stride rate for down-sampling
		'''
		super().__init__()
		padding = int((kSize - 1)/2)
		self.conv = nn.Conv3d(nIn, nOut, (kSize, kSize, kSize), stride=stride, padding=(padding, padding, padding), bias=False)

	def forward(self, input):
		'''
		:param input: input feature map
		:return: transformed feature map
		'''
		output = self.conv(input)
		return output

class CDilated(nn.Module):
	'''
	This class defines the dilated convolution.
	'''
	def __init__(self, nIn, nOut, kSize, stride=1, d=1):
		'''
		:param nIn: number of input channels
		:param nOut: number of output channels
		:param kSize: kernel size
		:param stride: optional stride rate for down-sampling
		:param d: optional dilation rate
		'''
		super().__init__()
		padding = int((kSize - 1)/2) * d
		self.conv = nn.Conv3d(nIn, nOut, (kSize, kSize, kSize), stride=stride, padding=(padding, padding, padding), bias=False, dilation=d)

	def forward(self, input):
		'''
		:param input: input feature map
		:return: transformed feature map
		'''
		output = self.conv(input)
		return output

class DilatedParllelResidualBlockB(nn.Module):
	'''
	This class defines the ESP block, which is based on the following principle
		Reduce ---> Split ---> Transform --> Merge
	'''
	def __init__(self, nIn=64, nOut=64, add=True):
		'''
		:param nIn: number of input channels
		:param nOut: number of output channels
		:param add: if true, add a residual connection through identity operation. You can use projection too as
				in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
				increase the module complexity
        '''
		super().__init__()
		n = int(nOut/5)
		n1 = nOut - 4*n
		self.c1 = C(nIn, n, 1, 1)
		self.d1 = CDilated(n, n1, 3, 1, 1) # dilation rate of 2^0
		self.d2 = CDilated(n, n, 3, 1, 2) # dilation rate of 2^1
		self.d4 = CDilated(n, n, 3, 1, 4) # dilation rate of 2^2
		self.d8 = CDilated(n, n, 3, 1, 8) # dilation rate of 2^3
		self.d16 = CDilated(n, n, 3, 1, 16) # dilation rate of 2^4
		self.bn = BR(nOut)
		self.add = add

	def forward(self, input):
		'''
		:param input: input feature map
		:return: transformed feature map
		'''
		# reduce
		output1 = self.c1(input)
		# split and transform
		d1 = self.d1(output1)
		d2 = self.d2(output1)
		d4 = self.d4(output1)
		d8 = self.d8(output1)
		d16 = self.d16(output1)

		# heirarchical fusion for de-gridding
		add1 = d2
		add2 = add1 + d4
		add3 = add2 + d8
		add4 = add3 + d16

		#merge
		combine = torch.cat([d1, add1, add2, add3, add4], 1)

		# if residual version
		if self.add:
			combine = input + combine
		output = self.bn(combine)
		return output
		
class PESP(nn.Module):
	def __init__(self, gt_mean, gt_std):
		super().__init__()
		self.gt_mean = torch.from_numpy(np.load(gt_mean)).float()
		self.gt_std = torch.from_numpy(np.load(gt_std)).float()

		self.iter_num = 3

		self.alpha = torch.nn.Parameter(torch.ones(1)*4)
		#self.alpha = torch.nn.Parameter(torch.ones(self.iter_num)*4)

		self.gen = nn.Sequential(
				nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
				nn.ReLU(inplace=True),
				self.make_layer(DilatedParllelResidualBlockB, 5),
				nn.Conv3d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv3d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv3d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
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

