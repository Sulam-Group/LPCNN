import os
import sys

import torch
from torch.utils import data

import numpy as np
import random

class QsmDataset(data.Dataset):

	def __init__(self, root, split='train', sep='partition', tesla=7, number=3, is_transform=True, augmentations=None, is_norm=False):
		self.root = root
		self.split = split
		self.is_transform = is_transform
		self.augmentations = augmentations
		self.is_norm = is_norm
		self.sep = sep
	
		self.tesla = tesla
		self.gamma = 42.57747892

		self.number = number

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
		self.input_list_file = os.path.join(self.root, split + '_phase' + str(self.number) + '.txt')
		self.dk_list_file = os.path.join(self.root, split + '_wdk' + str(self.number) + '.txt')
		self.mask_list_file = os.path.join(self.root, split + '_mask' + str(self.number) + '.txt')
		self.gt_list_file = os.path.join(self.root, split + '_gt' +  str(self.number)+ '.txt')

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

		random.seed(100)
		order = list(range(len(self.input_data)))
		random.shuffle(order)

		if self.sep == 'partition':
			train_amount = 2000
			val_amount = 500
		elif self.sep == 'whole':
			train_amount = 200
			val_amount = 20

		if split == 'train':
			self.input_data = [self.input_data[i] for i in order][:train_amount]
			self.mask_data = [self.mask_data[i] for i in order][:train_amount]
			self.gt_data = [self.gt_data[i] for i in order][:train_amount]
			self.dk_data = [self.dk_data[i] for i in order][:train_amount]

		elif split == 'validate':
			self.input_data = [self.input_data[i] for i in order][:val_amount]
			self.mask_data = [self.mask_data[i] for i in order][:val_amount]
			self.gt_data = [self.gt_data[i] for i in order][:val_amount]
			self.dk_data = [self.dk_data[i] for i in order][:val_amount]

	def __len__(self):
		return len(self.input_data)

	def __getitem__(self, index):
		
		input_path_list = self.input_data[index].split(' ')[:-1]
		input_name = input_path_list[0].split('/')[-1]
		mask_path_list = self.mask_data[index].split(' ')[:-1]

		gt_path = self.gt_data[index]
		dk_path_list = self.dk_data[index]

		x, y, z = np.load(input_path_list[0]).shape

		input_tensor_list = np.zeros((x, y, z, self.number))
		mask_tensor_list = np.zeros((x, y, z, self.number))

		gt_tensor = np.load(gt_path)

		for i in range(self.number):

			input_tensor_list[:, :, :, i] = np.load(input_path_list[i]) / (self.tesla*self.gamma)
			mask_tensor_list[:, :, :, i] = np.load(mask_path_list[i])

		if self.is_norm:	

			gt_tensor = gt_tensor - self.gt_mean
			gt_tensor = gt_tensor / self.gt_std
			gt_tensor = gt_tensor * mask_tensor_list[:, :, :, 0]

		input_tensor_list = input_tensor_list[np.newaxis, :, :, :, :]
		gt_tensor = gt_tensor[np.newaxis, :, :, :]
		#dk_tensor = dk_tensor[np.newaxis, :, :, :, np.newaxis]

		return input_tensor_list, gt_tensor, mask_tensor_list, dk_path_list, input_name
