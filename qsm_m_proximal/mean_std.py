import os
import sys
import argparse
import numpy as np

import torch
from torch.utils import data

from lib.utils import *
from lib.dataset.qsm_dataset import QsmDataset
from tqdm import tqdm
import socket

if socket.gethostname() == 'ka':
	global_path = '/home/kuowei/Desktop/'
elif socket.gethostname() == 'rafiki':	
	global_path = '/mnt/data0/kuowei/'
elif socket.gethostname() == 'kuowei':
	global_path = '/home/kuowei/Desktop/'
else:
	global_path = '/home-3/klai10@jhu.edu/data/kuowei/'

#dataset_path = global_path + '/qsm_dataset/qsm_B_r/mix16_data/partition/partition_data_list'
dataset_path = global_path + '/qsm_dataset/qsm_B_r/synthetic16_data/whole/data_list'

train_dataset = QsmDataset(dataset_path, split='train', is_norm=False)
train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

sq_data_sum = 0
data_sum = 0
total_num = 0

sq_ang_sum = np.zeros(3)
ang_sum = np.zeros(3)
number = 0

for batch_count, (input_data, gt_data, mask_data, _, input_name) in tqdm(enumerate(train_loader)):
	
	voxel_num = mask_data.squeeze(0).numpy().sum()

	mask = mask_data.squeeze(0).numpy()[np.newaxis, :, :, :]
	data = gt_data.squeeze(0).numpy()
	
	sq_data_sum += np.sum(np.square(data[mask==1]))
	data_sum += np.sum(data[mask==1])
	total_num += voxel_num

	#sq_ang_sum += np.square(ang_data.squeeze(0).numpy())
	#ang_sum += ang_data.squeeze(0).numpy()
	#number += 1

train_mean = data_sum / total_num
train_std = np.sqrt((sq_data_sum - total_num * np.square(train_mean)) / total_num)

#ang_mean = ang_sum / number
#ang_std = np.sqrt((sq_ang_sum - number * np.square(ang_mean)) / number)

#print(ang_sum)
#print(ang_mean)
#print(ang_std)
#np.save('train_ang_mean.npy', ang_mean)
#np.save('train_ang_std.npy', ang_std)
np.save('train_gt_mean.npy', train_mean)
np.save('train_gt_std.npy', train_std)
