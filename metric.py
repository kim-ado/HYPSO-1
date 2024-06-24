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

def norm_blocco(x):
    a = np.mean(x)
    c = np.std(x, ddof=1)
    if c == 0:
        c = 2.2204 * 10**(-16)
    y = ((x - a) / c) + 1
    return y, a, c

def onion_mult(onion1,onion2):

    N = onion1.size
    
    if (N > 1):      
        L = int(N/2)   
        
        a = onion1[0,0:L]
        b = onion1[0,L:onion1.shape[1]]
        b = np.concatenate(([b[0]],-b[1:b.shape[0]]))
        c = onion2[0,0:L]
        d = onion2[0,L:onion2.shape[1]]
        d = np.concatenate(([d[0]],-d[1:d.shape[0]]))
    
        if (N == 2):
            ris = np.concatenate((a*c-d*b, a*d+c*b))
        else:
            ris1 = onion_mult(np.reshape(a,(1,a.shape[0])),np.reshape(c,(1,c.shape[0])))
            ris2 = onion_mult(np.reshape(d,(1,d.shape[0])),np.reshape(np.concatenate(([b[0]],-b[1:b.shape[0]])),(1,b.shape[0])))
            ris3 = onion_mult(np.reshape(np.concatenate(([a[0]],-a[1:a.shape[0]])),(1,a.shape[0])),np.reshape(d,(1,d.shape[0])))
            ris4 = onion_mult(np.reshape(c,(1,c.shape[0])),np.reshape(b,(1,b.shape[0])))
    
            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
    
            ris = np.concatenate((aux1,aux2))
    else:
        ris = onion1 * onion2

    return ris

def onion_mult2D(onion1,onion2):

    N3 = onion1.shape[2]

    if (N3 > 1):
        L = int(N3/2)
        a = onion1[:,:,0:L]
        b = onion1[:,:,L:onion1.shape[2]]
        h = b[:,:,0]
        b = np.concatenate((h[:,:,None],-b[:,:,1:b.shape[2]]),axis=2)
        c = onion2[:,:,0:L]
        d = onion2[:,:,L:onion2.shape[2]]
        h = d[:,:,0]
        d = np.concatenate((h[:,:,None],-d[:,:,1:d.shape[2]]),axis=2)

        if (N3 == 2):
            ris = np.concatenate((a*c-d*b,a*d+c*b),axis=2)
        else:
            ris1 = onion_mult2D(a,c)
            h = b[:,:,0]
            ris2 = onion_mult2D(d,np.concatenate((h[:,:,None],-b[:,:,1:b.shape[2]]),axis=2))
            h = a[:,:,0]
            ris3 = onion_mult2D(np.concatenate((h[:,:,None],-a[:,:,1:a.shape[2]]),axis=2),d)
            ris4 = onion_mult2D(c,b)
            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = np.concatenate((aux1,aux2), axis=2)
    else:
        ris = onion1 * onion2   

    return ris

def onions_quality(dat1,dat2,size1):

    #dat1 = dat1.astype('float64')
    #dat2 = dat2.astype('float64')
    
    
    h = dat2[:,:,0]
    dat2 = np.concatenate((h[:,:,None],-dat2[:,:,1:dat2.shape[2]]),axis=2)
    
    N3 = dat1.shape[2]
    size2 = size1
    
    """ Block normalization """
    for i in range(N3):
      a1,s,t = norm_blocco(np.squeeze(dat1[:,:,i]))
      dat1[:,:,i] = a1
      
      if (s == 0):
          if (i == 0):
              dat2[:,:,i] = dat2[:,:,i] - s + 1
          else:
              dat2[:,:,i] = -(-dat2[:,:,i] - s + 1)
      else:
          if (i == 0):
              dat2[:,:,i] = (dat2[:,:,i] - s)/t + 1
          else:
              dat2[:,:,i] = -(((-dat2[:,:,i] - s)/t) + 1)    
    
    m1 = np.zeros((1,N3))
    m2 = np.zeros((1,N3))
    
    mod_q1m = 0
    mod_q2m = 0
    mod_q1 = np.zeros((size1,size2))
    mod_q2 = np.zeros((size1,size2))
    
    for i in range(N3):
        m1[0,i] = np.mean(np.squeeze(dat1[:,:,i]))
        m2[0,i] = np.mean(np.squeeze(dat2[:,:,i]))
        mod_q1m = mod_q1m + m1[0,i]**2
        mod_q2m = mod_q2m + m2[0,i]**2
        mod_q1 = mod_q1 + (np.squeeze(dat1[:,:,i]))**2
        mod_q2 = mod_q2 + (np.squeeze(dat2[:,:,i]))**2
    
    mod_q1m = np.sqrt(mod_q1m)
    mod_q2m = np.sqrt(mod_q2m)
    mod_q1 = np.sqrt(mod_q1)
    mod_q2 = np.sqrt(mod_q2)
    
    termine2 = mod_q1m * mod_q2m
    termine4 = mod_q1m**2 + mod_q2m**2
    int1 = (size1 * size2)/((size1 * size2)-1) * np.mean(mod_q1**2)
    int2 = (size1 * size2)/((size1 * size2)-1) * np.mean(mod_q2**2)
    termine3 = int1 + int2 - (size1 * size2)/((size1 * size2) - 1) * ((mod_q1m**2) + (mod_q2m**2))
    
    mean_bias = 2*termine2/termine4
    
    if (termine3==0):
        q = np.zeros((1,1,N3))
        q[:,:,N3-1] = mean_bias
    else:
        cbm = 2/termine3
        qu = onion_mult2D(dat1, dat2)
        qm = onion_mult(m1, m2)
        qv = np.zeros((1,N3))
        for i in range(N3):
            qv[0,i] = (size1 * size2)/((size1 * size2)-1) * np.mean(np.squeeze(qu[:,:,i]))
        q = qv - ((size1 * size2)/((size1 * size2) - 1.0)) * qm
        q = q * mean_bias * cbm
    return q


def q2n(I_GT, I_F, Q_blocks_size, Q_shift):

    N1 = I_GT.shape[0]
    N2 = I_GT.shape[1]
    N3 = I_GT.shape[2]
    
    size2 = Q_blocks_size
    
    stepx = int(np.ceil(N1/Q_shift))
    stepy = int(np.ceil(N2/Q_shift))
     
    if (stepy <= 0):
        stepy = 1
        stepx = 1
    
    est1 = (stepx - 1) * Q_shift + Q_blocks_size - N1
    print(est1)
    est2 = (stepy - 1) * Q_shift + Q_blocks_size - N2
    print(est2)
       
    if (est1 != 0) or (est2 != 0):
        refref = []
        fusfus = []
      
        for i in range(N3):
            a1 = np.squeeze(I_GT[:,:,0])
            
            ia1 = np.zeros((N1+est1,N2+est2))
            ia1[0:N1,0:N2] = a1
            ia1[:,N2:N2+est2] = ia1[:,N2-1:N2-est2-1:-1]
            ia1[N1:N1+est1,:] = ia1[N1-1:N1-est1-1:-1,:]

            if i == 0:
                refref = ia1
            elif i == 1:
                refref = np.concatenate((refref[:,:,None],ia1[:,:,None]),axis=2)
            else:
                refref = np.concatenate((refref,ia1[:,:,None]),axis=2)
            
            if (i < (N3-1)):
                I_GT = I_GT[:,:,1:I_GT.shape[2]]
        print("first loop")
        I_GT = refref
            
        for i in range(N3):
            a2 = np.squeeze(I_F[:,:,0])
            
            ia2 = np.zeros((N1+est1,N2+est2))
            ia2[0:N1,0:N2] = a2
            ia2[:,N2:N2+est2] = ia2[:,N2-1:N2-est2-1:-1]
            ia2[N1:N1+est1,:] = ia2[N1-1:N1-est1-1:-1,:]
            
            if i == 0:
                fusfus = ia2
            elif i == 1:
                fusfus = np.concatenate((fusfus[:,:,None],ia2[:,:,None]),axis=2)
            else:
                fusfus = np.concatenate((fusfus,ia2[:,:,None]),axis=2)
            
            if (i < (N3-1)):
                I_F = I_F[:,:,1:I_F.shape[2]]
            
        I_F = fusfus
    print("second loop")
      
    N1 = I_GT.shape[0]
    N2 = I_GT.shape[1]
    N3 = I_GT.shape[2]
    
    # Check if the number of bands is not a power of 2
    if ((int(np.ceil(np.log2(N3))) - np.log2(N3)) != 0):
        # Calculate the difference needed to make it a power of 2
        Ndif = (2**(np.ceil(np.log2(N3)))) - N3
        Ndif = int(Ndif)
        # Create an array of zeros with the required difference in shape
        dif = np.zeros((N1,N2,Ndif))
        # Add the zero padding to the original arrays
        I_GT = np.concatenate((I_GT, dif), axis = 2)
        I_F = np.concatenate((I_F, dif), axis = 2)
    
    N3 = I_GT.shape[2]

    print("Valori here")
    
    valori = np.zeros((stepx,stepy,N3))
    
    for j in range(stepx):
        for i in range(stepy):
            o = onions_quality(I_GT[ (j*Q_shift):(j*Q_shift)+Q_blocks_size, (i*Q_shift):(i*Q_shift)+size2,:], I_F[ (j*Q_shift):(j*Q_shift)+Q_blocks_size,(i*Q_shift):(i*Q_shift)+size2,:], Q_blocks_size)
            valori[j,i,:] = o
        
    Q2n_index_map = np.sqrt(np.sum(valori**2, axis=2))

    Q2n_index = np.mean(Q2n_index_map)
    
    return Q2n_index



def d_lam(blur, fused):
    """calculates Spectral Distortion Index (D_lambda).
    """
    dlam = q2n(blur, fused, 32, 32)

    return dlam 

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

def lsr(I_F, I_PAN, S=32):
    IHc = I_PAN.flatten()
    ILRc = I_F.reshape(-1, I_F.shape[2])

    w, _, _, _ = np.linalg.lstsq(ILRc, IHc, rcond=None)
    alpha = np.expand_dims(np.expand_dims(w, axis=0), axis=0)

    I_R = np.sum(I_F * np.tile(alpha, (I_F.shape[0], I_F.shape[1], 1)), axis=2)
    err_reg = I_PAN.flatten() - I_R.flatten()
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
	dsr = d_s_r(sharpened_cube, cube[:,:,sbi])
	b = (1-dsr)**beta
	return a*b, lam, dsr