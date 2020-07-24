
import os
import glob

data_path = 'data/numpy_data/whole'

train_set = ['Sub001', 'Sub002']
validation_set = ['Sub003']

if not os.path.exists('data/numpy_data/whole/list'):
	os.makedirs('data/numpy_data/whole/list')

with open('data/numpy_data/whole/list/train_gt.txt', 'w') as f:
	for subject in train_set:
		
		cosmos_dir = data_path + '/cosmos_data/' + subject + '/'
		phase_dir = data_path + '/phase_data/' + subject + '/'
	
		ori_num = len([name for name in os.listdir(phase_dir) if 'ori' in name])

		for ori in range(ori_num):
			f.write(cosmos_dir + subject + '_cosmos.npy\n')

with open('data/numpy_data/whole/list/train_mask.txt', 'w') as f:
	for subject in train_set:
		
		mask_dir = data_path + '/mask_data/' + subject + '/'

		ori_num = len([name for name in os.listdir(mask_dir) if 'ori' in name])

		for ori in range(ori_num):
			ori_name = 'ori' + str(ori+1)
			f.write(mask_dir + ori_name + '/' + subject + '_' + ori_name + '_mask.npy\n')

with open('data/numpy_data/whole/list/train_phase.txt', 'w') as f:
	for subject in train_set:
		
		phase_dir = data_path + '/phase_data/' + subject + '/'

		ori_num = len([name for name in os.listdir(phase_dir) if 'ori' in name])

		for ori in range(ori_num):
			ori_name = 'ori' + str(ori+1)
			f.write(phase_dir + ori_name + '/' + subject + '_' + ori_name + '_phase.npy\n')
with open('data/numpy_data/whole/list/train_dipole.txt', 'w') as f:
	for subject in train_set:

		dipole_dir = data_path + '/dipole_data/' + subject + '/'

		ori_num = len([name for name in os.listdir(dipole_dir) if 'ori' in name])

		for ori in range(ori_num):
			ori_name = 'ori' + str(ori+1)
			f.write(dipole_dir + ori_name + '/' + subject + '_' + ori_name + '_dipole.npy\n')

with open('data/numpy_data/whole/list/validation_gt.txt', 'w') as f:
	for subject in validation_set:
		
		cosmos_dir = data_path + '/cosmos_data/' + subject + '/'
		phase_dir = data_path + '/phase_data/' + subject + '/'
	
		ori_num = len([name for name in os.listdir(phase_dir) if 'ori' in name])

		for ori in range(ori_num):
			f.write(cosmos_dir + subject + '_cosmos.npy\n')

with open('data/numpy_data/whole/list/validation_mask.txt', 'w') as f:
	for subject in validation_set:
		
		mask_dir = data_path + '/mask_data/' + subject + '/'

		ori_num = len([name for name in os.listdir(mask_dir) if 'ori' in name])

		for ori in range(ori_num):
			ori_name = 'ori' + str(ori+1)
			f.write(mask_dir + ori_name + '/' + subject + '_' + ori_name + '_mask.npy\n')

with open('data/numpy_data/whole/list/validation_phase.txt', 'w') as f:
	for subject in validation_set:
		
		phase_dir = data_path + '/phase_data/' + subject + '/'

		ori_num = len([name for name in os.listdir(phase_dir) if 'ori' in name])

		for ori in range(ori_num):
			ori_name = 'ori' + str(ori+1)
			f.write(phase_dir + ori_name + '/' + subject + '_' + ori_name + '_phase.npy\n')
with open('data/numpy_data/whole/list/validation_dipole.txt', 'w') as f:
	for subject in validation_set:

		dipole_dir = data_path + '/dipole_data/' + subject + '/'

		ori_num = len([name for name in os.listdir(dipole_dir) if 'ori' in name])

		for ori in range(ori_num):
			ori_name = 'ori' + str(ori+1)
			f.write(dipole_dir + ori_name + '/' + subject + '_' + ori_name + '_dipole.npy\n')


