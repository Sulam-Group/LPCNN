
import os
import glob

import nibabel as nib
import numpy as np
import scipy.io
from pathlib import Path

'''

prepare training dataset.

'''

nifti_data_path = Path('data/nifti_data/')

numpy_data_path = Path('data/numpy_data')
numpy_data_path.mkdir(parents=True, exist_ok=True)

numpy_whole_data_path = numpy_data_path / 'whole'
numpy_whole_data_path.mkdir(parents=True, exist_ok=True)
numpy_partition_data_path = numpy_data_path / 'partition'
numpy_partition_data_path.mkdir(parents=True, exist_ok=True)

phase_data_list = list(nifti_data_path.glob('**/*phase.nii.gz'))
cosmos_data_list = list(nifti_data_path.glob('**/*cosmos.nii.gz'))
mask_data_list = list(nifti_data_path.glob('**/*mask.nii.gz'))
dipole_data_list = list(nifti_data_path.glob('**/*dipole.mat'))


for item in phase_data_list:

	subject = item.parts[-3]
	ori = item.parts[-2]

	directory = numpy_whole_data_path / 'phase_data' / subject / ori
	directory.mkdir(parents=True, exist_ok=True)

	phase = nib.load(str(item))
	phase_data = phase.get_fdata()
	np.save(str(directory / (subject + '_' + ori + '_phase.npy')), phase_data)
	print(subject + '_' + ori + '_phase saved.')

for item in cosmos_data_list:

	subject = item.parts[-3]

	directory = numpy_whole_data_path / 'cosmos_data' / subject
	directory.mkdir(parents=True, exist_ok=True) 

	cosmos = nib.load(str(item))
	cosmos_data = cosmos.get_fdata()
	np.save(str(directory / (subject + '_cosmos.npy')), cosmos_data)
	print(subject + '_cosmos saved.')

for item in mask_data_list:

	subject = item.parts[-3]
	ori = item.parts[-2]

	directory = numpy_whole_data_path / 'mask_data' / subject
	directory.mkdir(parents=True, exist_ok=True)

	mask = nib.load(str(item))
	mask_data = mask.get_fdata()
	np.save(str(directory / (subject + '_mask.npy')), mask_data)
	print(subject + '_mask saved.')

for item in dipole_data_list:

	subject = item.parts[-3]
	ori = item.parts[-2]

	directory = numpy_whole_data_path / 'dipole_data' / subject / ori
	directory.mkdir(parents=True, exist_ok=True)	

	dipole = scipy.io.loadmat(str(item))['C']
	dipole = np.swapaxes(dipole, 0, 1)
	
	np.save(str(directory / (subject + '_' + ori + '_dipole.npy')), dipole)
	print(subject + '_' + ori + '_dipole saved.')

