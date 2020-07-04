
import torch
import torch.nn.functional as F

def gaussian_new(window_size, dim, sigma=1.5, channel=1):
	
	center = float(window_size // 2)
	
	if dim == 1:
		x = torch.arange(-1*center, center+1)
		gauss = torch.exp(-(x**2)/(2*(sigma**2)))
		gaussian = gauss/gauss.sum()
		gaussian = gaussian.unsqueeze(0).unsqueeze(0)
		window = gaussian.expand(channel, 1, window_size).contiguous()

	elif dim == 2:
		x, y = torch.meshgrid([torch.arange(-1*center, center+1), torch.arange(-1*center, center+1)])
		gauss = torch.exp(-(x**2 + y**2)/(2*(sigma**2)))	
		gaussian = gauss/gauss.sum()
		gaussian = gaussian.unsqueeze(0).unsqueeze(0)
		window = gaussian.expand(channel, 1, window_size, window_size).contiguous()

	elif dim == 3:
		x, y, z = torch.meshgrid([torch.arange(-1*center, center+1), torch.arange(-1*center, center+1), torch.arange(-1*center, center+1)])
		gauss = torch.exp(-(x**2 + y**2 + z**2)/(2*(sigma**2)))
		gaussian = gauss/gauss.sum()
		gaussian = gaussian.unsqueeze(0).unsqueeze(0)
		window = gaussian.expand(channel, 1, window_size, window_size, window_size).contiguous()

	else:
		raise Exception('Unrecognized dimension.')

	return window

def _ssim(img1, img2, dim, window, window_size, channel, reduction='mean', dynamic_range=255, decomp=False):

	if dim == 2: 
		
		mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
		mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

		mu1_sq = mu1.pow(2)
		mu2_sq = mu2.pow(2)
		mu1_mu2 = mu1*mu2

		sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
		sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
		sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

		C1 = (0.01 * dynamic_range)**2
		C2 = (0.03 * dynamic_range)**2

		l = (2*mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
		cs = (2*sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
		ssim_map = l * cs

	elif dim == 3:

		mu1 = F.conv3d(img1, window, padding=window_size//2, groups=channel)
		mu2 = F.conv3d(img2, window, padding=window_size//2, groups=channel)

		mu1_sq = mu1.pow(2)
		mu2_sq = mu2.pow(2)
		mu1_mu2 = mu1*mu2

		sigma1_sq = F.conv3d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
		sigma2_sq = F.conv3d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
		sigma12 = F.conv3d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

		C1 = (0.01 * dynamic_range)**2
		C2 = (0.03 * dynamic_range)**2

		l = (2*mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
		cs = (2*sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
		ssim_map = l * cs

	else:
		raise Exception('Unrecognized dimension.')

	if reduction == 'mean':
		ssim = ssim_map.mean()
	elif reduction == 'sum':
		ssim = ssim_map.sum()
	elif reduction == 'none':
		ssim = ssim_map

	if decomp:
		return ssim, torch.mean(cs)
	else:
		return ssim

def ssim2d(img1, img2, sigma=1.5, window_size = 11, reduction='mean', dynamic_range=255, decomp=False):

	(_, channel, _, _) = img1.size()
	window = gaussian_new(window_size, 2, 1.5, channel)
	window = window.to(img1.device)
	window = window.type_as(img1)

	return _ssim(img1, img2, 2, window, window_size, channel, reduction, dynamic_range, decomp)

def ssim3d(img1, img2, sigma=1.5, window_size = 11, reduction='mean', dynamic_range=255, decomp=False):

	(_, channel, _, _, _) = img1.size()
	window = gaussian_new(window_size, 3, 1.5, channel)
	window = window.to(img1.device)
	window = window.type_as(img1)

	return _ssim(img1, img2, 3, window, window_size, channel, reduction, dynamic_range, decomp)

