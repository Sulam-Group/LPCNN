import os
import sys
import argparse
import numpy as np

import torch
from torch.utils import data

from lib.utils import *
from lib.dataset.qsm_dataset import QsmDataset
from tqdm import tqdm

global_path = '/home/kuowei/Desktop'
dataset_path = global_path + '/qsm_dataset/qsm_B_r/synthetic100_data/whole/data_list'

train_dataset = QsmDataset(dataset_path, split='train', is_norm=False)
train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

val_dataset = QsmDataset(dataset_path, split='validate', is_norm=False)
val_loader = data.DataLoader(val_dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

max_val = 0
min_val = 1


for batch_count, (input_data, gt_data, mask_data, ang_data, wt_data, input_name) in enumerate(train_loader):
	
	data = gt_data.squeeze(0).numpy()

	temp_max_gt = np.amax(data, axis=(1, 2, 3))
	temp_min_gt = np.amin(data, axis=(1, 2, 3))

	max_v = np.amax(np.column_stack((temp_max_gt, max_val)), axis=1)
	min_v = np.amin(np.column_stack((temp_min_gt, min_val)), axis=1)

	max_val = max_v
	min_val = min_v

for batch_count, (input_data, gt_data, mask_data, ang_data, wt_data, input_name) in enumerate(val_loader):
	
	data = gt_data.squeeze(0).numpy()

	temp_max_gt = np.amax(data, axis=(1, 2, 3))
	temp_min_gt = np.amin(data, axis=(1, 2, 3))
	
	max_v = np.amax(np.column_stack((temp_max_gt, max_val)), axis=1)
	min_v = np.amin(np.column_stack((temp_min_gt, min_val)), axis=1)

	max_val = max_v
	min_val = min_v


print(max_val)
print(min_val)

#max_val = np.amax(max_vec)
#min_val = np.amin(min_vec)
#print(max_val, min_val)

np.save('train_val_gt_max_val.npy', max_val)
np.save('train_val_gt_min_val.npy', min_val)

