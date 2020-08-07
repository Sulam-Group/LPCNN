import nibabel as nib
import numpy as np
import os

def qsm_display(qsm_data, original_nifti, mask, out_name='test_output'):
	print(original_nifti)
	orig_qsm = nib.load(original_nifti)
	orig_affine = orig_qsm.affine

	qsm_output = nib.Nifti1Image(qsm_data, orig_affine)
	qsm_output.to_filename(out_name + '_qsm.nii.gz')

	print(out_name + ' saved.')

	
