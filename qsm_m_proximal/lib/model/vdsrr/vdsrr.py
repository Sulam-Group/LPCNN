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

    def __init__(self, inplanes=64, planes=64):
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

		
class VDSRR(nn.Module):
	def __init__(self):
		super().__init__()
		self.residual_layer = self.make_layer(BasicBlock, 9)
		self.input = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
		self.output = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
		self.relu = nn.ReLU(inplace=True)
	
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
				m.weight.data.normal_(0, sqrt(2. / n))
				
	def make_layer(self, block, num_of_layer):
		layers = []
		for _ in range(num_of_layer):
			layers.append(block())
		return nn.Sequential(*layers)

	def forward(self, x):		 

		out = self.relu(self.input(x))
		out = self.residual_layer(out)
		out = self.output(out)

		return out
