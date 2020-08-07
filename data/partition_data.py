
import os
import glob

import numpy as np
from pathlib import Path

x = [(0, 64), (32, 96), (64, 128), (96, 160), (128, 192), (160, 224)]
y = [(0, 64), (32, 96), (64, 128), (96, 160), (128, 192), (160, 224)]
z1 = [(0, 64), (14, 78), (32, 96), (48, 112), (62, 126)]
z2 = [(0, 64), (14, 78), (32, 96), (46, 110)]

second_group = ['Sub005', 'Sub006', 'Sub008', 'Sub009']

whole_data_path = Path('data/numpy_data/whole')
partition_data_path = Path('data/numpy_data/partition')

mask_data_path = whole_data_path / 'mask_data'
mask_data_list = list(mask_data_path.glob('**/*.npy'))

for item in mask_data_list:

	subject = item.parts[-2]

	phase_path = whole_data_path / 'phase_data' / subject
	cosmos_path = whole_data_path / 'cosmos_data' / subject
	total_ori_num = len([name for name in phase_path.iterdir() if name.is_dir() and 'ori' in str(name)])

	if subject in second_group:
		z = z2
	else:
		z = z1

	for ori in range(1, total_ori_num+1):

		phase = np.load(str(phase_path / ('ori' + str(ori)) / (subject + '_ori' + str(ori) + '_phase.npy')))
		cosmos = np.load(str(cosmos_path / (subject + '_cosmos.npy')))
		mask = np.load(str(item))

		phase_dir = partition_data_path / 'phase_pdata' / subject / ('ori' + str(ori))
		phase_dir.mkdir(parents=True, exist_ok=True)

		cosmos_dir = partition_data_path / 'cosmos_pdata' / subject
		cosmos_dir.mkdir(parents=True, exist_ok=True)

		mask_dir = partition_data_path / 'mask_pdata' / subject
		mask_dir.mkdir(parents=True, exist_ok=True)

		number = 0

		for i_x in range(len(x)):
			for i_y in range(len(y)):
				for i_z in range(len(z)):
					valid_sum = np.sum(mask[x[i_x][0]:x[i_x][1], y[i_y][0]:y[i_y][1], z[i_z][0]:z[i_z][1]])
					if valid_sum > 32768:
						np.save(str(phase_dir / (subject + '_ori' + str(ori) + '_phase_p' + str(number))), phase[x[i_x][0]:x[i_x][1], y[i_y][0]:y[i_y][1], z[i_z][0]:z[i_z][1]])
						np.save(str(cosmos_dir / (subject + '_cosmos_p' + str(number))), cosmos[x[i_x][0]:x[i_x][1], y[i_y][0]:y[i_y][1], z[i_z][0]:z[i_z][1]])
						np.save(str(mask_dir / (subject + '_mask_p' + str(number))), mask[x[i_x][0]:x[i_x][1], y[i_y][0]:y[i_y][1], z[i_z][0]:z[i_z][1]])

						number += 1

		print(subject + '_ori' + str(ori) + '_partition saved!')

