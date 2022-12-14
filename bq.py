import numpy as np
from libsvm import svmutil  # used in brisque
from brisque import BRISQUE as bq
from math import isnan
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def scoreCube(cube: np.ndarray) -> np.ndarray:
    """Calculate the brisque metric.

    Args:
        cube (np.ndarray): The cube to be evaluated.

    Returns:
        np.ndarray: The brisque metric for each wavelength.
    """
    brisque_metric = np.zeros(cube.shape[2])
    for i in range(cube.shape[2]):
        brisque_metric[i] = scoreImage(cube[:, :, i])

    return brisque_metric


def scoreImage(image: np.ndarray) -> float:
    """Calculate the brisque metric.

    Args:
        image (np.ndarray): The image to be evaluated.

    Returns:
        float: The value in terms of the brisque metric.
    """

    s = bq()

    ret = s.get_score(image)

    if isnan(ret):
        ret = 100

    return ret
