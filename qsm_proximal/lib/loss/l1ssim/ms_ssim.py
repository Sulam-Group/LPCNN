
import torch
import torch.nn.functional as F

from lib.loss.l1ssim.ssim import ssim2d, ssim3d

def msssim(img1, img2, dim, levels=5, window_size=11, reduction='mean', dynamic_range=255, normalize=True):

	#sigmas = [0.5, 1, 2, 4, 8]	
	mssim = []
	mcs = []
	if dim == 2:

		for _ in range(levels):
			ssim, cs = ssim2d(img1, img2, 1.5, window_size=window_size, reduction=reduction, dynamic_range=dynamic_range, decomp=True)
			mssim.append(ssim)
			mcs.append(cs)

			img1 = F.avg_pool2d(img1, (2, 2))
			img2 = F.avg_pool2d(img2, (2, 2))

	elif dim == 3:

		for _ in range(levels):
			ssim, cs = ssim3d(img1, img2, 1.5, window_size=window_size, reduction=reduction, dynamic_range=dynamic_range, decomp=True)
			mssim.append(ssim)
			mcs.append(cs)

			img1 = F.avg_pool3d(img1, (2, 2, 2))
			img2 = F.avg_pool3d(img2, (2, 2, 2))

	mssim = torch.stack(mssim)
	mcs = torch.stack(mcs)

	# Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
	if normalize:
		mssim = (mssim + 1) / 2
		mcs = (mcs + 1) / 2

	# From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
	output = torch.prod(mcs[:-1]) * mssim[-1]

	return 1 - output
'''
a = torch.rand(1, 6, 32, 32, 32)
b = torch.rand(1, 6, 32, 32, 32)

b.requires_grad =True

optimizer = torch.optim.Adam([b], lr=0.01)

ssim_value = msssim(a, b, 3)

while ssim_value < 0.95:
	optimizer.zero_grad()
	ssim_out = -msssim(a, b, 3)
	ssim_value = - ssim_out.item()
	print(ssim_value)
	ssim_out.backward()
	optimizer.step()


print(msssim(a, b, 3))
'''
