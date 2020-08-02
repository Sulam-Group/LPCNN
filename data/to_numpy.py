
import os
import glob

import nibabel as nib
import numpy as np
import scipy.io

'''

prepare training dataset.

'''

nifti_data_path = 'data/nifti_data/'

if not os.path.exists('data/numpy_data'):
	os.makedirs('data/numpy_data/whole')
	os.makedirs('data/numpy_data/partition')

phase_data_list = []
cosmos_data_list = []
mask_data_list = []
dipole_data_list = []

for r, d, f in os.walk(nifti_data_path):
	for file in f:
		if 'phase.nii.gz' in file:
			phase_data_list.append(os.path.join(r, file))

for r, d, f in os.walk(nifti_data_path): 
	for file in f:
		if 'cosmos.nii.gz' in file:
			cosmos_data_list.append(os.path.join(r, file))

for r, d, f in os.walk(nifti_data_path):
	for file in f:
		if 'mask.nii.gz' in file:
			mask_data_list.append(os.path.join(r, file))

for r, d, f in os.walk(nifti_data_path): 
	for file in f:
		if 'dipole.mat' in file:
			dipole_data_list.append(os.path.join(r, file))

for item in phase_data_list:

	subject = item.split('/')[-3]
	ori = item.split('/')[-2]

	directory = 'data/numpy_data/whole/phase_data/' + subject + '/' + ori
	if not os.path.exists(directory):
		os.makedirs(directory)

	phase = nib.load(item)
	phase_data = phase.get_fdata()
	np.save(directory + '/' + subject + '_' + ori + '_phase.npy', phase_data)
	print(subject + '_' + ori + '_phase saved.')

for item in cosmos_data_list:

	subject = item.split('/')[-3]

	directory = 'data/numpy_data/whole/cosmos_data/' + subject
	if not os.path.exists(directory):
		os.makedirs(directory)

	cosmos = nib.load(item)
	cosmos_data = cosmos.get_fdata()
	np.save(directory + '/' + subject + '_cosmos.npy', cosmos_data)
	print(subject + '_cosmos saved.')

for item in mask_data_list:

	subject = item.split('/')[-3]
	ori = item.split('/')[-2]

	directory = 'data/numpy_data/whole/mask_data/' + subject
	if not os.path.exists(directory):
		os.makedirs(directory)

	mask = nib.load(item)
	mask_data = mask.get_fdata()
	np.save(directory + '/' + subject + '_mask.npy', mask_data)
	print(subject + '_mask saved.')

for item in dipole_data_list:

	subject = item.split('/')[-3]
	ori = item.split('/')[-2]

	directory = 'data/numpy_data/whole/dipole_data/' + subject + '/' + ori
	if not os.path.exists(directory):
		os.makedirs(directory)

	dipole = scipy.io.loadmat(item)['C']
	dipole = np.swapaxes(dipole, 0, 1)
	
	np.save(directory + '/' + subject + '_' + ori + '_dipole.npy', dipole)
	print(subject + '_' + ori + '_dipole saved.')

