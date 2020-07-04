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
#dataset_path = global_path + '/qsm_dataset/qsm_B_r/mix16_data/partition/partition_data_list'
dataset_path = global_path + '/qsm_dataset/qsm_B_r/mix5_data/whole/data_list1'

train_dataset = QsmDataset(dataset_path, split='validate', is_norm=False)
train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=False, drop_last=True)

total_sum = 0
total_num = 0

for batch_count, (input_data, gt_data, mask_data, _, input_name) in tqdm(enumerate(train_loader)):

	voxel_num = mask_data.squeeze(0).numpy().sum()

	mask = mask_data.squeeze(0).numpy().astype(int)[np.newaxis,:,:,:]
	data = gt_data.squeeze(0).numpy()

	mse = np.sum(np.square(data*mask)) / voxel_num

	total_sum += mse
	total_num += 1

total_mse = total_sum / total_num

print(total_mse)
