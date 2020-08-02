import os
import sys

import torch
from torch.utils import data

import numpy as np

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

		self.root_path = self.root + self.sep + '/list/'

		self.gt_mean = None
		self.gt_std = None

		if self.is_norm:

			gt_mean_name = 'train_gt_mean.npy'
			gt_std_name = 'train_gt_std.npy'

			self.gt_mean = np.load(os.path.join(self.root_path, gt_mean_name))
			self.gt_std = np.load(os.path.join(self.root_path, gt_std_name))

		#get data path
		self.data_list_file = os.path.join(self.root_path, split + '.txt')

		self.data_list = []
		
		with open(self.data_list_file, 'r') as f:
			for line in f:
				self.data_list.append(line.rstrip('\n'))

	def __len__(self):
		return len(self.data_list)

	def comp_convert(self, comp, data):

		if self.sep == 'partition':
			if data == 'phase':
				path_list = []
				for i in range(self.number):
					path_list.append(self.root + self.sep + '/phase_pdata/' + comp[0] + '/' + comp[i+2] + '/' + comp[0] + '_' + comp[i+2] + '_phase_' + comp[1] + '.npy')

			elif data == 'dipole':
				path_list = ''
				for i in range(self.number):
					path_list = path_list + self.root + 'whole' + '/dipole_data/' + comp[0] + '/' + comp[i+2] + '/' + comp[0] + '_' + comp[i+2] + '_dipole.npy '

			elif data == 'mask':
				path_list = self.root + self.sep + '/mask_pdata/' + comp[0] + '/' + comp[0] + '_mask_' + comp[1] + '.npy'

			elif data == 'gt':
				path_list = self.root + self.sep + '/cosmos_pdata/' + comp[0] + '/' + comp[0] + '_cosmos_' + comp[1] + '.npy'

			elif data == 'name':
				path_list = comp[0] + '_'
				for i in range(self.number):
					path_list = path_list + comp[i+2]
				path_list = path_list + '_' + comp[1]

		elif self.sep == 'whole':
			if data == 'phase':
				path_list = []
				for i in range(self.number):
					path_list.append(self.root + self.sep + '/phase_data/' + comp[0] + '/' + comp[i+1] + '/' + comp[0] + '_' + comp[i+1] + '_phase.npy')
			
			elif data == 'dipole':
				path_list = ''
				for i in range(self.number):
					path_list = path_list + self.root + 'whole' + '/dipole_data/' + comp[0] + '/' + comp[i+1] + '/' + comp[0] + '_' + comp[i+1] + '_dipole.npy '

			elif data == 'mask':
				path_list = self.root + self.sep + '/mask_data/' + comp[0] + '/' + comp[0] + '_mask.npy'

			elif data == 'gt':
				path_list = self.root + self.sep + '/cosmos_data/' + comp[0] + '/' + comp[0] + '_cosmos.npy'

			elif data == 'name':
				path_list = comp[0] + '_'
				for i in range(self.number):
					path_list = path_list + comp[i+1]
		
		return path_list


	def __getitem__(self, index):
		
		data_comp_list = self.data_list[index].split(' ')

		phase_path_list = self.comp_convert(data_comp_list, 'phase')
		dipole_path_list = self.comp_convert(data_comp_list, 'dipole')
		mask_path = self.comp_convert(data_comp_list, 'mask') 
		gt_path = self.comp_convert(data_comp_list, 'gt')
		
		data_name = self.comp_convert(data_comp_list, 'name')

		x, y, z = np.load(phase_path_list[0]).shape

		phase_tensor_list = np.zeros((x, y, z, self.number))

		gt_tensor = np.load(gt_path)
		mask_tensor = np.load(mask_path)

		for i in range(self.number):

			phase_tensor_list[:, :, :, i] = np.load(phase_path_list[i]) / (self.tesla*self.gamma)

		if self.is_norm:	

			gt_tensor = gt_tensor - self.gt_mean
			gt_tensor = gt_tensor / self.gt_std
			gt_tensor = gt_tensor * mask_tensor

		phase_tensor_list = phase_tensor_list[np.newaxis, :, :, :, :]
		gt_tensor = gt_tensor[np.newaxis, :, :, :]

		return phase_tensor_list, gt_tensor, mask_tensor, dipole_path_list, data_name
