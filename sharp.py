import os

import matplotlib.pyplot as plt
import numpy as np
import spectral.io.envi as envi

import bq
import h1data


class sharp:
    def __init__(self):
        self.initial_cube = None
        self.refrence_cube = None

        self.sharpened_cube = None
        self.sharpest_band_index = None

        # metrics - no refrence
        self.brisque = {}

        # metrics - full refrence
        self.sam = {}

    def sharpen_cs(self):
        """Sharpen the cube using component substitution method."""
        self.sharpened_cube = component_subtitution(
            self.initial_cube, self.sharpest_band_index)

    @classmethod
    def fromh1data(cls, h1data: h1data):
        """Create a sharp object from a h1data object.

        Args:
            h1data (h1data): h1data object.

        Returns:
            sharp: sharp object.
        """
        sharp_obj = cls()

        if h1data.l1a_cube is None:
            try:
                h1data.calibrate_cube()
            except:
                sharp_obj.initial_cube = h1data.raw_cube
        else:
            sharp_obj.initial_cube = h1data.l1a_cube

        # Initialise the sharpest band index to the middle band
        sharp_obj.sharpest_band_index = np.shape(
            sharp_obj.initial_cube)[2] // 2

        return sharp_obj

    @classmethod
    def fromenvifile(cls, headerfile_path: str, datafile_path: str = None):
        """Create a sharp object from a envi file.

        Args:
            path (str): Path to the envi file.

        Returns:
            sharp: sharp object.
        """
        sharp_obj = cls()

        if ".hdr" not in headerfile_path:
            raise ValueError("Path to envi header file must end with .hdr")
        elif datafile_path is None:
            # assume data file is in same folder as header file
            try:
                datafile_path = headerfile_path.replace(".hdr", ".bip")
            except:
                raise ValueError(
                    "Function expects data to be in .bip format")

        temp_obj = envi.open(headerfile_path, datafile_path)
        sharp_obj.initial_cube = temp_obj.load()

        return sharp_obj

    def get_brisque(self) -> np.ndarray:

        self.brisque["initial"] = bq.scoreCube(self.initial_cube)

        if self.refrence_cube is not None:
            self.brisque["refrence"] = bq.scoreCube(self.initial_cube)

        if self.sharpened_cube is not None:
            self.brisque["sharpend"] = bq.scoreCube(self.initial_cube)

        return self.brisque

    def get_sam(self) -> np.ndarray:
        """Get the sam metric.

        Returns:
            float: The sam metric.
        """
        self.sam = sam(self.initial_cube, self.refrence_cube)
        return self.sam

    def load_refrence_cube(self, refrence_cube: np.ndarray) -> np.ndarray:
        """Load a refrence cube.

        Args:
            refrence_cube (np.ndarray): The refrence cube.

        Returns:
            np.ndarray: The refrence cube.
        """
        self.refrence_cube = refrence_cube
        return self.refrence_cube

    def evaluate(self) -> None:
        """Evaluate the sharpness of the initial cube.

        Args:
            refrence_cube (np.ndarray): The refrence cube.
        """

        if isinstance(self.initial_cube, np.ndarray) == False:
            raise ValueError("No initial cube loaded")
        elif isinstance(self.refrence_cube, np.ndarray) == False:
            self.brisque = self.get_brisque()
        else:
            self.brisque = self.get_brisque()
            self.sam = self.get_sam()


def sam(image: np.ndarray, refrence_image: np.ndarray) -> np.ndarray:
    """Calculate the spectral angle (SAM) metric.

    Args:
        image (np.ndarray): The image to be evaluated.
        refrence_image (np.ndarray): The refrence image.

    Returns:
        np.ndarray: The sam metric for each wavelength.
    """
    # check if image and refrence_image have same shape
    if image.shape != refrence_image.shape:
        raise ValueError("image and refrence_image must have same shape")

    sam_metric = np.zeros(image.shape[2])
    for i in range(image.shape[2]):
        # TODO: add sam score for each wavelength
        sam_metric[i] = np.mean((image[:, :, i]) + refrence_image[:, :, i])

    return sam_metric


def component_subtitution(image: np.ndarray, sharpest_band_index: int = None) -> np.ndarray:
    """Perform component substitution on an image.

    Args:
        image (np.ndarray): The image to be sharpened.
        sharpest_band_index (int, optional): The sharpest band. Defaults to None.

    Returns:
        np.ndarray: The sharpened image.
    """
    if sharpest_band_index is None:
        sharpest_band_index = image.shape[2] // 2

    # TODO: do PCA on image
    # TODO: histogram match first PCA component to the sharpest band
    # TODO: replace first component with the sharpest band

    sharpened_image = np.zeros(image.shape)
    for i in range(image.shape[2]):
        sharpened_image[:, :, i] = image[:, :, sharpest_band_index]

    return sharpened_image


def main():
    print("Hello World")
    return


if __name__ == "__main__":
    main()
