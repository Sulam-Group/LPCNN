
import os
import glob
import argparse
from pathlib import Path

whole_data_path = Path('data/numpy_data/whole')
partition_data_path = Path('data/numpy_data/partition')

train_set = ['Sub001', 'Sub002', 'Sub004', 'Sub005', 'Sub008', 'Sub009']
validation_set = ['Sub003', 'Sub006']

def combination(total_ori_num, number, comb_list=None, comb=None, previous_x=0):
	if comb_list is None:
		comb_list = []
	if comb is None:
		comb = ''	

	if number >= 1:
		for x in range(previous_x+1, total_ori_num+1):
			new_comb = comb + str(x) + ' '
			combination(total_ori_num, number - 1, comb_list, new_comb, x)
	else:
		comb_list.append(comb[:-1])	

	return comb_list

def create_list(args):

	whole_list_path = whole_data_path / 'list'
	whole_list_path.mkdir(parents=True, exist_ok=True)

	whole_train_path = whole_list_path / 'train.txt'
	with whole_train_path.open(mode='w') as f:
		for subject in train_set:
		
			input_dir = whole_data_path / 'phase_data' / subject

			total_ori_num = len([name for name in input_dir.iterdir() if name.is_dir() and 'ori' in str(name)])
			comb_list = combination(total_ori_num, args.number)
			comb_num = len(comb_list)

			for comb in comb_list:
				ori_list = comb.split(' ')

				f.write(subject + ' ')
				for ori in ori_list[:-1]:
					f.write('ori' + ori + ' ')
				f.write('ori' + ori_list[-1] + '\n')

	whole_validation_path = whole_list_path / 'validation.txt'
	with whole_validation_path.open(mode='w') as f:
		for subject in validation_set:
		
			input_dir = whole_data_path / 'phase_data' / subject

			total_ori_num = len([name for name in input_dir.iterdir() if name.is_dir() and 'ori' in str(name)])
			comb_list = combination(total_ori_num, args.number)
			comb_num = len(comb_list)

			for comb in comb_list:
				ori_list = comb.split(' ')

				f.write(subject + ' ')
				for ori in ori_list[:-1]:
					f.write('ori' + ori + ' ')
				f.write('ori' + ori_list[-1] + '\n')


	partition_list_path = partition_data_path / 'list'
	partition_list_path.mkdir(parents=True, exist_ok=True)

	partition_train_path = partition_list_path / 'train.txt'
	with partition_train_path.open(mode='w') as f:
		for subject in train_set:
		
			input_dir = partition_data_path / 'phase_pdata' / subject

			total_ori_num = len([name for name in input_dir.iterdir() if name.is_dir() and 'ori' in str(name)])
			comb_list = combination(total_ori_num, args.number)
			comb_num = len(comb_list)

			sub_input_dir = input_dir / 'ori1'
			for comb in comb_list:
				ori_list = comb.split(' ')
	
				num = len([name for name in sub_input_dir.iterdir() if '.npy' in str(name)])	
				for k in range(num):
					f.write(subject + ' p' + str(k) + ' ')
					for ori in ori_list[:-1]:
						f.write('ori' + ori + ' ')
					f.write('ori' + ori_list[-1] + '\n')

	partition_validation_path = partition_list_path / 'validation.txt'
	with partition_validation_path.open(mode='w') as f:
		for subject in validation_set:
		
			input_dir = partition_data_path / 'phase_pdata' / subject

			total_ori_num = len([name for name in input_dir.iterdir() if name.is_dir() and 'ori' in str(name)])
			comb_list = combination(total_ori_num, args.number)
			comb_num = len(comb_list)

			sub_input_dir = input_dir / 'ori1'
			for comb in comb_list:
				ori_list = comb.split(' ')
		
				num = len([name for name in sub_input_dir.iterdir() if '.npy' in str(name)])
				for k in range(num):
					f.write(subject + ' p' + str(k) + ' ')
					for ori in ori_list[:-1]:
						f.write('ori' + ori + ' ')
					f.write('ori' + ori_list[-1] + '\n')


parser = argparse.ArgumentParser(description='dataset list')
parser.add_argument('--number', type=int, default=1, choices=[1, 2, 3, 4, 5], help='input number')

if __name__ == '__main__':
	args = parser.parse_args()
	create_list(args)
