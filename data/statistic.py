
import os
import glob

import numpy as np
from pathlib import Path

whole_train = Path('data/numpy_data/whole/list/train.txt')
whole_validation = Path('data/numpy_data/whole/list/validation.txt')

partition_train = Path('data/numpy_data/partition/list/train.txt')
partition_validation = Path('data/numpy_data/partition/list/validation.txt')

range_max = 0
range_min = 1

with whole_train.open(mode='r') as f:

	for line in f:
		subject = line.strip('\n').split(' ')[0]
		cosmos_path = Path('data/numpy_data/whole/cosmos_data/' + subject + '/' + subject + '_cosmos.npy')
		data = np.load(str(cosmos_path))
		temp_max = np.amax(data)
		temp_min = np.amin(data)
		
		range_max = max(temp_max, range_max)
		range_min = min(temp_min, range_min)

with whole_validation.open(mode='r') as f:

	for line in f:
		subject = line.strip('\n').split(' ')[0]
		cosmos_path = Path('data/numpy_data/whole/cosmos_data/' + subject + '/' + subject + '_cosmos.npy')
		data = np.load(str(cosmos_path))
		temp_max = np.amax(data)
		temp_min = np.amin(data)
		
		range_max = max(temp_max, range_max)
		range_min = min(temp_min, range_min)

np.save(str(Path('data/numpy_data/whole/list/train_val_gt_max.npy')), range_max)
np.save(str(Path('data/numpy_data/partition/list/train_val_gt_max.npy')), range_max)
np.save(str(Path('data/numpy_data/whole/list/train_val_gt_min.npy')), range_min)
np.save(str(Path('data/numpy_data/partition/list/train_val_gt_min.npy')), range_min)

whole_sq_data_sum = 0
whole_data_sum = 0
whole_total_num = 0

with whole_train.open(mode='r') as f:
	
	for line in f:
		subject = line.strip('\n').split(' ')[0]

		data = np.load(str(Path('data/numpy_data/whole/cosmos_data/' + subject + '/' + subject + '_cosmos.npy')))
		
		mask = np.load(str(Path('data/numpy_data/whole/mask_data/' + subject + '/' + subject + '_mask.npy')))

		voxel_num = mask.sum()

		whole_sq_data_sum += np.sum(np.square(data[mask==1]))
		whole_data_sum += np.sum(data[mask==1])
		whole_total_num += voxel_num	
		
whole_train_mean = whole_data_sum / whole_total_num
whole_train_std = np.sqrt((whole_sq_data_sum - whole_total_num * np.square(whole_train_mean)) / whole_total_num)

np.save(str(Path('data/numpy_data/whole/list/train_gt_mean.npy')), whole_train_mean)
np.save(str(Path('data/numpy_data/whole/list/train_gt_std.npy')), whole_train_std)

partition_sq_data_sum = 0
partition_data_sum = 0
partition_total_num = 0

with partition_train.open(mode='r') as f:
	
	for line in f:
		subject = line.strip('\n').split(' ')[0]
		partition = line.strip('\n').split(' ')[1]

		data = np.load(str(Path('data/numpy_data/partition/cosmos_pdata/' + subject + '/' + subject + '_cosmos_' + partition + '.npy')))
		
		mask = np.load(str(Path('data/numpy_data/partition/mask_pdata/' + subject + '/' + subject + '_mask_' + partition + '.npy')))

		voxel_num = mask.sum()

		partition_sq_data_sum += np.sum(np.square(data[mask==1]))
		partition_data_sum += np.sum(data[mask==1])
		partition_total_num += voxel_num	
		
partition_train_mean = partition_data_sum / partition_total_num
partition_train_std = np.sqrt((partition_sq_data_sum - partition_total_num * np.square(partition_train_mean)) / partition_total_num)

np.save(str(Path('data/numpy_data/partition/list/train_gt_mean.npy')), partition_train_mean)
np.save(str(Path('data/numpy_data/partition/list/train_gt_std.npy')), partition_train_std)

whole_total_sum = 0
whole_total_num = 0

with whole_validation.open(mode='r') as f:
	
	for line in f:
		subject = line.strip('\n').split(' ')[0]

		data = np.load(str(Path('data/numpy_data/whole/cosmos_data/' + subject + '/' + subject + '_cosmos.npy')))
		
		mask = np.load(str(Path('data/numpy_data/whole/mask_data/' + subject + '/' + subject + '_mask.npy')))

		voxel_num = mask.sum()
		
		mse = np.sum(np.square(data*mask)) / voxel_num

		whole_total_sum += mse
		whole_total_num += 1
		
whole_mse = whole_total_sum / whole_total_num

np.save(str(Path('data/numpy_data/whole/list/validation_mse.npy')), whole_mse)

partition_total_sum = 0
partition_total_num = 0

with partition_validation.open(mode='r') as f:
	
	for line in f:
		subject = line.strip('\n').split(' ')[0]
		partition = line.strip('\n').split(' ')[1]

		data = np.load(str(Path('data/numpy_data/partition/cosmos_pdata/' + subject + '/' + subject + '_cosmos_' + partition + '.npy')))
		
		mask = np.load(str(Path('data/numpy_data/partition/mask_pdata/' + subject + '/' + subject + '_mask_' + partition + '.npy')))

		voxel_num = mask.sum()
		
		mse = np.sum(np.square(data*mask)) / voxel_num

		partition_total_sum += mse
		partition_total_num += 1
		
partition_mse = partition_total_sum / partition_total_num

np.save(str(Path('data/numpy_data/partition/list/validation_mse.npy')), partition_mse)

	
