import numpy as np
import scipy.ndimage as ft
from skimage.transform.integral import integral_image as integral
from math import ceil, floor, log2
from skimage.util import view_as_blocks
from scipy.ndimage import uniform_filter
from PIL import Image

def qnr(cube: np.ndarray, sharpened_cube, sbi: int, r, alpha=1, beta=1, p=1, q=1, ws=7):
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
    lam = d_lambda(sharpened_cube, cube, p=p)
    dro = d_rho(sharpened_cube, cube[:, :, sbi], r)
    
    a = (1-lam)**alpha
    b = (1-dro)**beta
    print("Finished QNR")
    return a*b, lam, dro


def d_lambda(sharp_cube: np.ndarray, cube: np.ndarray, p=1):
    """calculates Spectral Distortion Index (D_lambda).

    :param ms: low resolution multispectral image.
    :param fused: high resolution fused image.
    :param p: parameter to emphasize large spectral differences (default = 1).

    :returns:  float -- D_lambda.
    """
    L = cube.shape[2]

    M1 = np.ones((L, L))
    M2 = np.ones((L, L))

    for l in range(L):
        for r in range(l + 1, L):
            M1[l, r] = M1[r, l] = uqi(sharp_cube[:, :, l], sharp_cube[:, :, r])
            M2[l, r] = M2[r, l] = uqi(cube[:, :, l], cube[:, :, r])
     
    print("finished d_lambda")
    diff = np.abs(M1 - M2)**p
    return (1./(L*(L-1)) * np.sum(diff))**(1./p)


def ERGAS(outputs, labels, ratio):
    """
        Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS).


        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144
        [Ranchin00]         T. Ranchin and L. Wald, "Fusion of high spatial and spectral resolution images: the ARSIS concept and its implementation,"
                            Photogrammetric Engineering and Remote Sensing, vol. 66, no. 1, pp. 4961, January 2000.
        [Vivone20]          G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods",
                            IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.

        Parameters
        ----------
        outputs : Numpy Array
            The Fused image. Dimensions: H, W, Bands
        labels : Numpy Array
            The reference image. Dimensions: H, W, Bands
        ratio : int
            PAN-MS resolution ratio

        Return
        ------
        ergas_index : float
            The ERGAS index.

    """
    eps = 1e-20
    mu = np.mean(labels[:,:,4], axis=(0, 1)) ** 2
    nbands = labels.shape[-1]
    error = np.mean((outputs[:,:,4] - labels[:,:,4]) ** 2, axis=(0, 1))
    ergas_index = 100 * ratio * np.sqrt(np.sum(error / mu + eps) / nbands)

    return np.mean(ergas_index).item()

def d_rho(fused_cube: np.ndarray, sharpest_band_cube: np.ndarray, sigma):
    """
        Spatial Quality Index based on local cross-correlation.

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144

        Parameters
        ----------
        outputs : Numpy Array
            Fused MultiSpectral img. Dimensions: H, W, Bands
        pan : Numpy Array
            Panchromatic img. Dimensions: H, W, Bands
        sigma : int
            The windows size on which calculate the Drho index; Accordingly with the paper it should be the
            resolution scale which elapses between MS and PAN.

        Return
        ------
        d_rho : float
            The d_rho index

    """
    half_width = ceil(sigma / 2)
    rho = np.clip(local_cross_correlation(fused_cube, sharpest_band_cube, half_width), a_min=-1.0, a_max=1.0)
    d_rho = 1.0 - rho
    print("finished d_rho")
    return np.mean(d_rho).item()



def local_cross_correlation(img_1, img_2, half_width):
    """
        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144

        Cross-Correlation Field computation.

        Parameters
        ----------
        img_1 : Numpy Array
            First image on which calculate the cross-correlation. Dimensions: H, W
        img_2 : Numpy Array
            Second image on which calculate the cross-correlation. Dimensions: H, W
        half_width : int
            The semi-size of the window on which calculate the cross-correlation


        Return
        ------
        L : Numpy array
            The cross-correlation map between img_1 and img_2

    """

    w = int(half_width)
    ep = 1e-20

    if (len(img_1.shape)) != 3:
        img_1 = np.expand_dims(img_1, axis=-1)
    if (len(img_2.shape)) != 3:
        img_2 = np.expand_dims(img_2, axis=-1)

    img_1 = np.pad(img_1.astype(np.float64), ((w, w), (w, w), (0, 0)))
    img_2 = np.pad(img_2.astype(np.float64), ((w, w), (w, w), (0, 0)))

    img_1_cum = np.zeros(img_1.shape)
    img_2_cum = np.zeros(img_2.shape)
    for i in range(img_1.shape[-1]):
        img_1_cum[:, :, i] = integral(img_1[:, :, i]).astype(np.float64)
    for i in range(img_2.shape[-1]):
        img_2_cum[:, :, i] = integral(img_2[:, :, i]).astype(np.float64)

    img_1_mu = (img_1_cum[2 * w:, 2 * w:, :] - img_1_cum[:-2 * w, 2 * w:, :] - img_1_cum[2 * w:, :-2 * w, :]
                + img_1_cum[:-2 * w, :-2 * w, :]) / (4 * w ** 2)
    img_2_mu = (img_2_cum[2 * w:, 2 * w:, :] - img_2_cum[:-2 * w, 2 * w:, :] - img_2_cum[2 * w:, :-2 * w, :]
                + img_2_cum[:-2 * w, :-2 * w, :]) / (4 * w ** 2)

    img_1 = img_1[w:-w, w:-w, :] - img_1_mu
    img_2 = img_2[w:-w, w:-w, :] - img_2_mu

    img_1 = np.pad(img_1.astype(np.float64), ((w, w), (w, w), (0, 0)))
    img_2 = np.pad(img_2.astype(np.float64), ((w, w), (w, w), (0, 0)))

    i2 = img_1 ** 2
    j2 = img_2 ** 2
    ij = img_1 * img_2

    i2_cum = np.zeros(i2.shape)
    j2_cum = np.zeros(j2.shape)
    ij_cum = np.zeros(ij.shape)

    for i in range(i2_cum.shape[-1]):
        i2_cum[:, :, i] = integral(i2[:, :, i]).astype(np.float64)
    for i in range(j2_cum.shape[-1]):
        j2_cum[:, :, i] = integral(j2[:, :, i]).astype(np.float64)
    for i in range(ij_cum.shape[-1]):
        ij_cum[:, :, i] = integral(ij[:, :, i]).astype(np.float64)

    sig2_ij_tot = (ij_cum[2 * w:, 2 * w:, :] - ij_cum[:-2 * w, 2 * w:, :] - ij_cum[2 * w:, :-2 * w, :]
                   + ij_cum[:-2 * w, :-2 * w, :])
    sig2_ii_tot = (i2_cum[2 * w:, 2 * w:, :] - i2_cum[:-2 * w, 2 * w:, :] - i2_cum[2 * w:, :-2 * w, :]
                   + i2_cum[:-2 * w, :-2 * w, :])
    sig2_jj_tot = (j2_cum[2 * w:, 2 * w:, :] - j2_cum[:-2 * w, 2 * w:, :] - j2_cum[2 * w:, :-2 * w, :]
                   + j2_cum[:-2 * w, :-2 * w, :])

    sig2_ii_tot = np.clip(sig2_ii_tot, ep, sig2_ii_tot.max())
    sig2_jj_tot = np.clip(sig2_jj_tot, ep, sig2_jj_tot.max())

    xcorr = sig2_ij_tot / ((sig2_ii_tot * sig2_jj_tot) ** 0.5 + ep)
    

    return xcorr


def _uqi_single(GT, P, ws):
    N = ws**2
    window = np.ones((ws, ws))

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
    return np.mean(q_map[s:-s, s:-s])

def uqi(GT, P, ws=8):
    """calculates universal image quality index (uqi).

    :param GT: first (original) input image.
    :param P: second (deformed) input image.
    :param ws: sliding window size (default = 8).

    :returns:  float -- uqi value.
    """
    GT,P = _initial_check(GT, P)
    return np.mean([_uqi_single(GT[:, :, i],P[:, :, i], ws) for i in range(GT.shape[2])])




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

    return GT.astype(np.float64),P.astype(np.float64)


