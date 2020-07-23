import numpy as np
import torch
import torch.nn as nn
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

		
class PNDNN(nn.Module):
	def __init__(self):
		super().__init__()

		self.alpha = torch.nn.Parameter(torch.ones(1))

		self.gen = nn.Sequential(
				nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
				nn.ReLU(inplace=True),
				self.make_layer(BasicBlock, 4),
				nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
		)
		#self.input = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
		#self.residual_layer = self.make_layer(BasicBlock, 1)
		#self.conv1 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
		#self.conv2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
		#self.conv3 = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
		#self.relu = nn.ReLU(inplace=True)
	
		#for m in self.modules():
		#	if isinstance(m, nn.Conv3d):
		#		n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
		#		m.weight.data.normal_(0, sqrt(2. / n))
				
	def make_layer(self, block, num_of_layer):
		layers = []
		for _ in range(num_of_layer):
			layers.append(block())
		return nn.Sequential(*layers)

	def forward(self, y, dk, mask):		 

		iter_num = 5

		out = []

		x_est = self.alpha * torch.irfft(dk * torch.rfft(y, 3, normalized=True, onesided=False), 3, normalized=True, onesided=False)
		pn_x_pred = 0

		for i in range(iter_num):
			x_input = x_est + pn_x_pred
			x_pred = self.gen(x_input)

			pn_x_pred = x_pred - self.alpha * torch.irfft(dk * dk * torch.rfft(x_pred, 3, normalized=True, onesided=False), 3, normalized=True, onesided=False)
			
			out.append(x_pred)
	
		return out
