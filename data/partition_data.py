
import os
import glob

import numpy as np


mask_data_list = []

x = [(0, 64), (32, 96), (64, 128), (96, 160), (128, 192), (160, 224)]
y = [(0, 64), (32, 96), (64, 128), (96, 160), (128, 192), (160, 224)]
z1 = [(0, 64), (14, 78), (32, 96), (48, 112), (62, 126)]
z2 = [(0, 64), (14, 78), (32, 96), (46, 110)]

second_group = ['Sub005', 'Sub006', 'Sub008', 'Sub009']

for r, d, f in os.walk('data/numpy_data/whole/mask_data'):
	for file in f:
		if '.npy' in file:
			mask_data_list.append(os.path.join(r, file))

for item in mask_data_list:

	subject = item.split('/')[-2]

	total_ori_num = len([name for name in os.listdir('data/numpy_data/whole/phase_data/' + subject) if 'ori' in name])

	if subject in second_group:
		z = z2
	else:
		z = z1

	for ori in range(1, total_ori_num+1):

		phase = np.load('data/numpy_data/whole/phase_data/' + subject + '/ori' + str(ori) + '/' + subject + '_ori' + str(ori) + '_phase.npy')
		cosmos = np.load('data/numpy_data/whole/cosmos_data/' + subject + '/' + subject + '_cosmos.npy')
		mask = np.load(item)

		phase_dir = 'data/numpy_data/partition/phase_pdata/' + subject + '/ori' + str(ori)
		if not os.path.exists(phase_dir):
			os.makedirs(phase_dir)

		cosmos_dir = 'data/numpy_data/partition/cosmos_pdata/' + subject
		if not os.path.exists(cosmos_dir):
			os.makedirs(cosmos_dir)

		mask_dir = 'data/numpy_data/partition/mask_pdata/' + subject
		if not os.path.exists(mask_dir):
			os.makedirs(mask_dir)

		number = 0

		for i_x in range(len(x)):
			for i_y in range(len(y)):
				for i_z in range(len(z)):
					valid_sum = np.sum(mask[x[i_x][0]:x[i_x][1], y[i_y][0]:y[i_y][1], z[i_z][0]:z[i_z][1]])
					if valid_sum > 32768:
						np.save(phase_dir + '/' + subject + '_ori' + str(ori) + '_phase_p' + str(number), phase[x[i_x][0]:x[i_x][1], y[i_y][0]:y[i_y][1], z[i_z][0]:z[i_z][1]])
						np.save(cosmos_dir + '/' + subject + '_cosmos_p' + str(number), cosmos[x[i_x][0]:x[i_x][1], y[i_y][0]:y[i_y][1], z[i_z][0]:z[i_z][1]])
						np.save(mask_dir + '/' + subject + '_mask_p' + str(number), mask[x[i_x][0]:x[i_x][1], y[i_y][0]:y[i_y][1], z[i_z][0]:z[i_z][1]])

						number += 1

		print(subject + '_ori' + str(ori) + '_partition saved!')

