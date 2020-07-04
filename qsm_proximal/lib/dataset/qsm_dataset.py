import os
import sys

import torch
from torch.utils import data

import numpy as np


class QsmDataset(data.Dataset):

	def __init__(self, root, split='train', tesla=7, is_transform=True, augmentations=None, is_norm=False):
		self.root = root
		self.split = split
		self.is_transform = is_transform
		self.augmentations = augmentations
		self.is_norm = is_norm

		self.tesla = tesla
		self.gamma = 42.57747892

		#self.input_mean = None
		#self.input_std = None
		self.gt_mean = None
		self.gt_std = None

		if self.is_norm:

			#input_mean_name = 'train_input_mean.npy'
			#input_std_name = 'train_input_std.npy'

			gt_mean_name = 'train_gt_mean.npy'
			gt_std_name = 'train_gt_std.npy'

			#self.input_mean = np.load(os.path.join(self.root, input_mean_name))
			#self.input_std = np.load(os.path.join(self.root, input_std_name))

			self.gt_mean = np.load(os.path.join(self.root, gt_mean_name))
			self.gt_std = np.load(os.path.join(self.root, gt_std_name))

		#get data path
		self.input_list_file = os.path.join(self.root, split + '_phase.txt')
		self.dk_list_file = os.path.join(self.root, split + '_wdk.txt')
		self.mask_list_file = os.path.join(self.root, split + '_mask.txt')
		self.gt_list_file = os.path.join(self.root, split + '_gt.txt')

		self.input_data = []
		self.mask_data = []
		self.gt_data = []
		self.dk_data = []
		
		with open(self.input_list_file, 'r') as f:
			for line in f:
				self.input_data.append(line.rstrip('\n'))
		with open(self.mask_list_file, 'r') as f:
			for line in f:
				self.mask_data.append(line.rstrip('\n'))
		with open(self.gt_list_file, 'r') as f:
			for line in f:
				self.gt_data.append(line.rstrip('\n'))
		with open(self.dk_list_file, 'r') as f:
			for line in f:
				self.dk_data.append(line.rstrip('\n'))

	def __len__(self):
		return len(self.input_data)

	def __getitem__(self, index):
		
		input_path = self.input_data[index]
		input_name = input_path.split('/')[-1]
		mask_path = self.mask_data[index]

		gt_path = self.gt_data[index]
		dk_path = self.dk_data[index]

		input_tensor = np.load(input_path) / (self.tesla*self.gamma)
		mask_tensor = np.load(mask_path)
		gt_tensor = np.load(gt_path)
		#dk_tensor = np.load(dk_path)

		if self.is_norm:	
			#input_tensor = input_tensor - self.input_mean
			#input_tensor = input_tensor / self.input_std
			#input_tensor = input_tensor * mask_tensor

			gt_tensor = gt_tensor - self.gt_mean
			gt_tensor = gt_tensor / self.gt_std
			gt_tensor = gt_tensor * mask_tensor
			#gt_tensor = np.tanh(gt_tensor*10)
	
		input_tensor = input_tensor[np.newaxis, :, :, :]
		gt_tensor = gt_tensor[np.newaxis, :, :, :]
		#dk_tensor = dk_tensor[np.newaxis, :, :, :, np.newaxis]

		return input_tensor, gt_tensor, mask_tensor, dk_path, input_name
