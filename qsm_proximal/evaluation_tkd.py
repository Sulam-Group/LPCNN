import os
import sys

import numpy as np
import nibabel as nib

from lib.utils import *



# parameter for relative value evaluation(validate)
## mix_data
#whole_validate_mse = 0.0007885226196501292

whole_validate_mse = 0.0007885226196501292
whole_validate_rmse = np.sqrt(whole_validate_mse)

root_dir = '/home/kuowei/Desktop/'

test_set = 'val'

gt_path = root_dir + 'qsm_dataset/qsm_B_r/real5_data/whole/cosmos_data/'
mask_path = root_dir + 'qsm_dataset/qsm_B_r/real5_data/whole/mask_data/'

test_path = '/home/kuowei/Desktop/MICCAI_result/starQSM/'
#test_path = '/home/kuowei/Desktop/qsm_dl/qsm_w_ori/vis_output/vdsrr_ud_test_Bmodel/'
# directory


#input_data_std_path = '/home/kuowei/Desktop/qsm_dataset/qsm_B_r/mix16_data/whole/data_list/train_input_std.npy'
#input_data_mean_path = '/home/kuowei/Desktop/qsm_dataset/qsm_B_r/mix16_data/whole/data_list/train_input_mean.npy'

#gt_data_std_path = '/home/kuowei/Desktop/qsm_dataset/qsm_B_r/mix16_data/whole/data_list/train_gt_std.npy'
#gt_data_mean_path = '/home/kuowei/Desktop/qsm_dataset/qsm_B_r/mix16_data/whole/data_list/train_gt_mean.npy'

#subject_list = ['Sub001', 'Sub002', 'Sub003', 'Sub004', 'Sub005', 'Sub006', 'Sub008', 'Sub009']
subject_list = ['Sub001']

mse_loss = 0
sq_mse_loss = 0

ssim_perf = 0	
sq_ssim_perf = 0

w_psnr_perf = 0
sq_w_psnr_perf = 0

r_psnr_perf = 0
sq_r_psnr_perf = 0

number = 0

for i in range(len(subject_list)):

	if i <= 3:
		j_len = 1
	else:
		j_len = 4

	for j in range(j_len):

		test_img = nib.load(test_path + subject_list[i] + '_ori' + str(j+1) + '_starqsm.nii.gz')
		test_data = test_img.get_fdata()[:, :, :, np.newaxis]

		gt_data = np.load(gt_path + subject_list[i] + '/' + subject_list[i] + '_cosmos.npy')

		mask = np.load(mask_path + subject_list[i] + '/ori' + str(j+1) + '/' +  subject_list[i] + '_ori' + str(j+1) + '_mask.npy')[:,:,:,np.newaxis]

		og_gt = gt_data[:, :, :, np.newaxis] * mask
	
		#mse_loss += qsm_mse(og_gt, test_data, mask, roi=True)
		#sq_mse_loss += np.square(qsm_mse(og_gt, test_data, mask, roi=True))

		mse_loss += np.sqrt(qsm_mse(og_gt, test_data, mask, roi=True))
		sq_mse_loss += np.square(np.sqrt(qsm_mse(og_gt, test_data, mask, roi=True)))

		ssim_perf += qsm_ssim(og_gt, test_data, mask, root_dir)	
		sq_ssim_perf += np.square(qsm_ssim(og_gt, test_data, mask, root_dir))

		w_psnr_perf += qsm_psnr(og_gt, test_data, mask, root_dir, roi=False)
		sq_w_psnr_perf += np.square(qsm_psnr(og_gt, test_data, mask, root_dir, roi=False))

		r_psnr_perf += qsm_psnr(og_gt, test_data, mask, root_dir, roi=True)
		sq_r_psnr_perf += np.square(qsm_psnr(og_gt, test_data, mask, root_dir, roi=True))
		print(qsm_psnr(og_gt, test_data, mask, root_dir, roi=True))
		number += 1



avg_mse_loss = mse_loss / number
std_mse_loss = np.sqrt((sq_mse_loss - number * np.square(avg_mse_loss)) / number)

avg_ssim_perf = ssim_perf / number
std_ssim_perf = np.sqrt((sq_ssim_perf - number * np.square(avg_ssim_perf)) / number)

avg_w_psnr_perf = w_psnr_perf / number
std_w_psnr_perf = np.sqrt((sq_w_psnr_perf - number * np.square(avg_w_psnr_perf)) / number)

avg_r_psnr_perf = r_psnr_perf / number
std_r_psnr_perf = np.sqrt((sq_r_psnr_perf - number * np.square(avg_r_psnr_perf)) / number)

print('##Test Mse: %.8f+-%.8f PSNR: %.8f+-%.8f(%.8f+-%.8f) SSIM: %.8f+-%.8f' %(avg_mse_loss/whole_validate_rmse, std_mse_loss/whole_validate_rmse, avg_w_psnr_perf, std_w_psnr_perf, avg_r_psnr_perf, std_r_psnr_perf, avg_ssim_perf, std_ssim_perf))





























