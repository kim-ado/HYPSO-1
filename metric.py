import torch
import numpy as np
import warnings
from scipy.ndimage import uniform_filter
eps = 1e-10

#Cross-correlation matrix
def cross_correlation(H_fuse, H_ref):
    N_spectral = H_fuse.shape[2]

    # Reshaping fused and reference data
    H_fuse_reshaped = H_fuse.reshape(-1, N_spectral)
    H_ref_reshaped = H_ref.reshape(-1, N_spectral)

    # Calculating mean value
    mean_fuse = np.mean(H_fuse_reshaped, axis=0)
    mean_ref = np.mean(H_ref_reshaped, axis=0)

    # Calculating numerator and denominator for cross-correlation
    numerator = np.sum((H_fuse_reshaped - mean_fuse) * (H_ref_reshaped - mean_ref), axis=0)
    denominator = np.sqrt(np.sum((H_fuse_reshaped - mean_fuse)**2, axis=0) * np.sum((H_ref_reshaped - mean_ref)**2, axis=0))

    # Cross-correlation for each spectral band
    CC = numerator / denominator
    # Average cross-correlation over all spectral bands
    CC = np.mean(CC)
    
    return CC

# Spectral-Angle-Mapper (SAM)
def sam(H_fuse, H_ref):
    # Reshape fused and reference data to (Channels, Height*Width)
    H_fuse_reshaped = H_fuse.reshape(-1, H_fuse.shape[-1]).T
    H_ref_reshaped = H_ref.reshape(-1, H_ref.shape[-1]).T
    
    # Calculate inner product
    inner_prod = np.nansum(H_fuse_reshaped * H_ref_reshaped, axis=0)
    fuse_norm = np.sqrt(np.nansum(H_fuse_reshaped**2, axis=0))
    ref_norm = np.sqrt(np.nansum(H_ref_reshaped**2, axis=0))
    
    # Calculate SAM
    cos_theta = inner_prod / (fuse_norm * ref_norm)
    cos_theta = np.clip(cos_theta, -1, 1)  # Ensure values are in [-1, 1]
    SAM = np.rad2deg(np.nanmean(np.arccos(cos_theta)))
    return SAM

def psnr (P,GT,MAX=None):
	"""calculates peak signal-to-noise ratio (psnr).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param MAX: maximum value of datarange (if None, MAX is calculated using image dtype).

	:returns:  float -- psnr value in dB.
	"""
	if MAX is None:
		MAX = MAX = np.max(GT)

	GT,P = _initial_check(GT,P)

	mse_value = mse(GT,P)
	if mse_value == 0.:
		return np.inf
	return 10 * np.log10(MAX**2 /mse_value)

def rmse (P,GT):
	"""calculates root mean squared error (rmse).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.

	:returns:  float -- rmse value.
	"""
	GT,P = _initial_check(GT,P)
	return np.sqrt(mse(GT,P))

def mse (GT,P):
	"""calculates mean squared error (mse).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.

	:returns:  float -- mse value.
	"""
	GT,P = _initial_check(GT,P)
	return np.mean((GT.astype(np.float64)-P.astype(np.float64))**2)


def ergas(H_fuse, H_ref, beta):
    # Reshape images to (Channels, Height*Width)
    H_fuse_reshaped = H_fuse.reshape(-1, H_fuse.shape[-1]).T
    H_ref_reshaped = H_ref.reshape(-1, H_ref.shape[-1]).T
    N_pixels = H_fuse_reshaped.shape[1]
    
    # Calculate RMSE of each band
    rmse = np.sqrt(np.nansum((H_ref_reshaped - H_fuse_reshaped)**2, axis=1) / N_pixels)
    mu_ref = np.mean(H_ref_reshaped, axis=1)
    
    # Calculate ERGAS
    ergas = 100 * (1 / beta**2) * np.sqrt(np.nansum((rmse / mu_ref)**2) / H_fuse_reshaped.shape[0])
    return ergas


def _uqi_single(GT,P,ws):
	N = ws**2
	window = np.ones((ws,ws))

	GT_sq = GT*GT
	P_sq = P*P
	GT_P = GT*P

	GT_sum = uniform_filter(GT, ws)    
	P_sum =  uniform_filter(P, ws)     
	GT_sq_sum = uniform_filter(GT_sq, ws)  
	P_sq_sum = uniform_filter(P_sq, ws)  
	GT_P_sum = uniform_filter(GT_P, ws)

	GT_P_sum_mul = GT_sum*P_sum
	GT_P_sum_sq_sum_mul = GT_sum*GT_sum + P_sum*P_sum
	numerator = 4*(N*GT_P_sum - GT_P_sum_mul)*GT_P_sum_mul
	denominator1 = N*(GT_sq_sum + P_sq_sum) - GT_P_sum_sq_sum_mul
	denominator = denominator1*GT_P_sum_sq_sum_mul

	q_map = np.ones(denominator.shape)
	index = np.logical_and((denominator1 == 0) , (GT_P_sum_sq_sum_mul != 0))
	q_map[index] = 2*GT_P_sum_mul[index]/GT_P_sum_sq_sum_mul[index]
	index = (denominator != 0)
	q_map[index] = numerator[index]/denominator[index]

	s = int(np.round(ws/2))
	return np.mean(q_map[s:-s,s:-s])

def uqi (GT,P,ws=2):
	"""calculates universal image quality index (uqi).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param ws: sliding window size (default = 8).

	:returns:  float -- uqi value.
	"""
	GT,P = _initial_check(GT,P)
	return np.mean([_uqi_single(GT[:,:,i],P[:,:,i],ws) for i in range(GT.shape[2])])

def _initial_check(GT,P):
	assert GT.shape == P.shape, "Supplied images have different sizes " + \
	str(GT.shape) + " and " + str(P.shape)
	if GT.dtype != P.dtype:
		msg = "Supplied images have different dtypes " + \
			str(GT.dtype) + " and " + str(P.dtype)
		warnings.warn(msg)
	

	if len(GT.shape) == 2:
		GT = GT[:,:,np.newaxis]
		P = P[:,:,np.newaxis]

	return GT,P

def d_s (pan,fused,q=1):
	"""calculates Spatial Distortion Index (D_S).

	:param pan: high resolution panchromatic image.
	:param ms: low resolution multispectral image.
	:param fused: high resolution fused image.
	:param q: parameter to emphasize large spatial differences (default = 1).
	:param r: ratio of high resolution to low resolution (default=4).
	:param ws: sliding window size (default = 7).

	:returns:  float -- D_S.
	"""
	pan = pan.astype(np.float64)
	fused = fused.astype(np.float64)

	L = fused.shape[2]

	M1 = np.zeros(L)

	for l in range(L):
		M1[l] = uqi(fused[:,:,l],pan)

	diff = np.abs(M1)**q
	return ((1./L)*(np.sum(diff)))**(1./q)

def d_lambda (ms,fused,p=1):
	"""calculates Spectral Distortion Index (D_lambda).

	:param ms: low resolution multispectral image.
	:param fused: high resolution fused image.
	:param p: parameter to emphasize large spectral differences (default = 1).

	:returns:  float -- D_lambda.
	"""
	L = ms.shape[2]

	M1 = np.ones((L,L))
	M2 = np.ones((L,L))

	for l in range(L):
		for r in range(l+1,L):
			M1[l,r] = M1[r,l] = uqi(fused[:,:,l],fused[:,:,r])
			M2[l,r] = M2[r,l] = uqi(ms[:,:,l],ms[:,:,r])

	diff = np.abs(M1 - M2)**p
	return (1./(L*(L-1)) * np.sum(diff))**(1./p)

def d_s_r(I_F, I_PAN):
    cd = lsr(I_F, I_PAN)
    Ds_R_index = 1 - cd
    return Ds_R_index

def lsr(I_F, I_PAN):
    IHc = I_PAN.flatten()
    ILRc = I_F.reshape(-1, I_F.shape[2])

    # Multivariate linear regression
    w, _, _, _ = np.linalg.lstsq(ILRc, IHc, rcond=None)
    alpha = w.reshape((1, 1, -1))

    # Fitted least squares intensity
    I_R = np.sum(I_F * alpha, axis=2)

    # Space-varying least squares error
    err_reg = I_PAN.flatten() - I_R.flatten()

    # Coefficient of determination
    cd = 1 - (np.var(err_reg) / np.var(I_PAN.flatten()))

    return cd


def qnr(cube: np.ndarray, sharpened_cube, sbi: int, alpha=1, beta=1, p=1):	
	"""calculates Quality with No Reference (QNR).
	
	:param pan: high resolution panchromatic image.
	:param ms: low resolution multispectral image.
	:param fused: high resolution fused image.
	:param alpha: emphasizes relevance of spectral distortions to the overall.
	:param beta: emphasizes relevance of spatial distortions to the overall.
	:param p: parameter to emphasize large spectral differences (default = 1).
	:param q: parameter to emphasize large spatial differences (default = 1).
	:param r: ratio of high resolution to low resolution (default=4).
	:param ws: sliding window size (default = 7).

	:returns:  float -- QNR.
	"""

	lam = d_lambda(cube,sharpened_cube,p=p)
	a = (1-lam)**alpha
	dsr = d_s_r(sharpened_cube, sharpened_cube[:,:,sbi])
	b = (1-dsr)**beta
	return a*b, lam, dsr