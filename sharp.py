import numpy as np
import spectral.io.envi as envi

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

        self.name = "sharp"

    def set_name(self, name: str):
        self.name = name

    def sharpen(self, method: str = "cs") -> None:
        """Sharpen the cube method. The method is set by the method argument.
        """
        if method == "cs":
            self.sharpened_cube = component_subtitution(
                self.initial_cube, self.sharpest_band_index)
        else:
            errmsg = "Method not supported."
            errmsg += "\n - Supported methods are: 'cs'"
            raise ValueError(errmsg)

    @ classmethod
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

        sharp_obj.sharpest_band_index = h1data.center_wavelength

        return sharp_obj

    @ classmethod
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
        import bq

        self.brisque["initial"] = bq.scoreCube(self.initial_cube)

        if self.refrence_cube is not None:
            self.brisque["refrence"] = bq.scoreCube(self.refrence_cube)

        if self.sharpened_cube is not None:
            self.brisque["sharpend"] = bq.scoreCube(self.sharpened_cube)

        return self.brisque

    def get_sam(self) -> np.ndarray:
        """Get the sam metric.

        Returns:
            float: The sam metric.
        """
        self.sam = sam(self.initial_cube, self.refrence_cube)
        return self.sam

    def set_sharpest_band_index(self, band_index: int) -> int:
        """Set the sharpest band index.

        Args:
            band_index (int): The band index.

        Returns:
            int: The band index.
        """
        self.sharpest_band_index = band_index
        return self.sharpest_band_index

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
            # self.sam = self.get_sam()


def sam(image: np.ndarray, refrence_image: np.ndarray, return_image: bool = False) -> np.ndarray:
    """Calculate the spectral angle (SAM) metric.

    Args:
        image (np.ndarray): The image to be evaluated.
        refrence_image (np.ndarray): The refrence image.
        return_image (bool, optional): Return the sam image. Defaults to False.

    Returns:
        np.ndarray: The sam metric for each wavelength.

    """
    # check if image and refrence_image have same shape
    if image.shape != refrence_image.shape:
        raise ValueError("image and refrence_image must have same shape")

    sam_image = np.zeros([image.shape[0], image.shape[1]])
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            vector_img = image[x, y, :].flatten()
            vector_ref = refrence_image[x, y, :].flatten()

            angle = np.arccos(np.dot(vector_img, vector_ref) /
                              (np.linalg.norm(vector_img) * np.linalg.norm(vector_ref)))
            sam_image[x, y] = angle

    avg_sam = np.mean(sam_image)
    if return_image:
        return avg_sam, sam_image
    else:
        return avg_sam


def component_subtitution(image: np.ndarray, sharpest_band_index: int = None) -> np.ndarray:
    """Perform component substitution on an image cube.

    Args:
        image (np.ndarray) [NxMxL]: The image cube to be sharpened.
                                        N = first spatial dimension
                                        M = second spatial dimension
                                        L = number of bands

        sharpest_band_index (int, optional): The sharpest band. Defaults to center band.

    Returns:
        np.ndarray: The sharpened image cube.
    """
    from numpy.linalg import svd
    from skimage.exposure import match_histograms

    if sharpest_band_index is None:
        sharpest_band_index = image.shape[2] // 2

    # Do PCA on image
    # inspo: https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
    # Reshape image to 2D array with each row representing samples
    img_variable_form = np.reshape(
        image, (image.shape[0] * image.shape[1], image.shape[2]))
    U, S, Vh = svd(img_variable_form, full_matrices=False)
    principal_components = np.dot(U, np.diag(S))
    component_cube = np.reshape(
        principal_components, (image.shape[0], image.shape[1], image.shape[2]))

    # Match histogram of sharpest band to first component
    matched_sharpest_band = match_histograms(
        image[:, :, sharpest_band_index], component_cube[:, :, 0])

    # Replace first component with matched sharpest band
    fixed_component_cube = component_cube
    fixed_component_cube[:, :, 0] = matched_sharpest_band
    fixed_cc_variable_form = np.reshape(
        fixed_component_cube, (image.shape[0] * image.shape[1], image.shape[2]))

    # Do inverse PCA
    sharpend_variable_form = np.dot(fixed_cc_variable_form, Vh)
    sharpend_cube = np.reshape(
        sharpend_variable_form, (image.shape[0], image.shape[1], image.shape[2]))

    return sharpend_cube


def wavelet_sharpen(base_image: np.ndarray,
                    reference_image: np.ndarray,
                    wavelet: str = "db1",
                    level: int = 1,
                    provide_coeffs: bool = False
                    ) -> np.ndarray:
    """Perform wavelet sharpening on a single image using a reference image and a wavelet transform.

    Args:
        base_image (np.ndarray): The base image.
        reference_image (np.ndarray): The reference image.
        wavelet (str, optional): The wavelet to use. Defaults to "db1".
        level (int, optional): The level of the wavelet transform. Defaults to 1.
        provide_coeffs (bool, optional): Whether to return the wavelet coefficients. Defaults to False.

    Returns:
        np.ndarray: The sharpened image.
    """
    import pywt

    # Perform wavelet transform on base image
    base_coeffs = pywt.wavedec2(base_image, wavelet, level=level)

    # Perform wavelet transform on reference image
    reference_coeffs = pywt.wavedec2(reference_image, wavelet, level=level)

    # Replace base image coefficients with reference image coefficients
    sharpened_coeffs = reference_coeffs
    sharpened_coeffs[0] = base_coeffs[0]

    # Perform inverse wavelet transform
    sharpened_image = pywt.waverec2(base_coeffs, wavelet)

    if provide_coeffs:
        return sharpened_image, base_coeffs, reference_coeffs
    else:
        return sharpened_image


class SharpeningAlg:
    def __init__(self, type: str,
                 mother_wavelet: str = None,
                 wavelet_level: int = None,
                 strategy: str = None,
                 filter_order: int = None,):

        self.type = type

        def check_wavelet(mother_wavelet):
            if mother_wavelet is None:
                raise ValueError("mother_wavelet must be specified")
            else:
                return mother_wavelet

        def check_wavelet_level(wavelet_level):
            if wavelet_level is None or wavelet_level <= 0:
                raise ValueError(
                    "wavelet_level must be specified as a positive integer")
            else:
                return wavelet_level

        def check_strategy(strategy):
            valid_strategies = [
                "regular",
                "ladder"
            ]

            if strategy is None and (type == "wavelet" or type == "laplacian"):
                raise ValueError(f"strategy must be specified for type {type}")
            elif strategy.lower() not in valid_strategies:
                raise ValueError(
                    f"strategy: {strategy}, must be one of the following: {valid_strategies}")
            return strategy.lower()

        def check_filter_order(filter_order):
            if filter_order is None:
                raise ValueError("filter_order must be specified")
            elif filter_order <= 0:
                raise ValueError("filter_order must be an integer")
            else:
                return filter_order

        if type == "wavelet":
            self.mother_wavelet = check_wavelet(mother_wavelet)
            self.wavelet_level = check_wavelet_level(wavelet_level)
            self.strategy = check_strategy(strategy)
        elif type == "laplacian":
            self.strategy = check_strategy(strategy)
            self.filter_order = check_filter_order(filter_order)
        elif type == "cs":
            pass
        elif type == "none":
            pass
        else:
            raise ValueError("type must be one of the following: {}".format(
                ["wavelet", "laplacian", "cs", "none"]))

    def sharpen(self, cube: np.array, sbi: int) -> np.array:
        """ Sharpen a cube using the specified sharpening algorithm.

            Args:
            --------
                cube (np.ndarray): The cube to sharpen.
                sbi (np.ndarray): The Sharpest Base Image Index in the cube.

            Returns:
            --------
                np.ndarray: The sharpened cube.
        """

        if self.type == "wavelet":
            return self.wavelet_cube_sharpen(cube, sbi)
        elif self.type == "laplacian":
            return self.laplacian_cube_sharpen(cube, sbi)
        elif self.type == "cs":
            return self.cs_sharpen(cube, sbi)
        elif self.type == "none":
            return cube
        else:
            raise ValueError("type must be one of the following: {}".format(
                ["wavelet", "laplacian", "cs", "none"]))

    def wavelet_cube_sharpen(self, cube: np.ndarray, sbi: int) -> np.ndarray:
        sharpened_cube = np.zeros(cube.shape)
        if self.strategy == "regular":
            sharpened_cube = wavelet_cube_sharpen_regular(
                cube, sbi, self.mother_wavelet, self.wavelet_level)
        elif self.strategy == "ladder":
            sharpened_cube = cube
        return sharpened_cube

    def laplacian_cube_sharpen(self, cube: np.ndarray, sbi: int) -> np.ndarray:
        sharpened_cube = np.zeros(cube.shape)
        if self.strategy == "regular":
            sharpened_cube = laplacian_cube_sharpen_regular(
                cube, sbi, self.filter_order)
        elif self.strategy == "ladder":
            sharpened_cube = cube
        return sharpened_cube

    def cs_sharpen(self, cube: np.ndarray, sbi: int = None) -> np.ndarray:
        """Sharpen an image using the component substitution algorithm.
        """
        return component_subtitution(cube, sbi)


def wavelet_cube_sharpen_regular(cube: np.ndarray, sbi: int, mother_wavelet: str, wavelet_level: int) -> np.ndarray:
    """Sharpen a cube using the wavelet sharpening algorithm with a regular strategy.

    Args:
        cube (np.ndarray): The cube to sharpen.
        sbi (int): The Sharpest Base Image Index in the cube.
        mother_wavelet (str): The mother wavelet to use.
        wavelet_level (int): The level of the wavelet transform.

    Returns:
        np.ndarray: The sharpened cube.
    """
    sharpened_cube = np.zeros(cube.shape)
    for i in range(cube.shape[2]):
        sharpened_cube[:, :, i] = wavelet_sharpen(
            cube[:, :, i], cube[:, :, sbi], mother_wavelet, wavelet_level)

    return sharpened_cube


def wavelet_cube_sharpen_ladder(cube: np.ndarray, sbi: int, mother_wavelet: str, wavelet_level: int) -> np.ndarray:
    """Sharpen a cube using the wavelet sharpening algorithm with a ladder strategy.

    Args:
        cube (np.ndarray): The cube to sharpen.
        sbi (int): The Sharpest Base Image Index in the cube.
        mother_wavelet (str): The mother wavelet to use.
        wavelet_level (int): The level of the wavelet transform.

    Returns:
        np.ndarray: The sharpened cube.
    """
    sharpened_cube = np.zeros(cube.shape)

    # TODO: Implement ladder strategy

    return sharpened_cube


def laplacian_cube_sharpen_regular(cube: np.ndarray, sbi: int, filter_order: int = 5) -> np.ndarray:
    """Sharpen a cube using the laplacian sharpening algorithm with a regular strategy.

    Args:
        cube (np.ndarray): The cube to sharpen.
        sbi (int): The Sharpest Base Image Index in the cube.
        filter_order (int): The size of the kernel to use.

    Returns:
        np.ndarray: The sharpened cube.
    """
    import numpy as np
    from skimage.filters import butterworth
    from skimage.exposure import match_histograms

    sharpened_cube = np.zeros(cube.shape)

    for i in range(cube.shape[2]):
        ref_im = cube[:, :, sbi]
        base_im = cube[:, :, i]
        ref_im = np.multiply((ref_im - np.mean(ref_im)) , (np.std(base_im) / np.std(ref_im))) + np.mean(base_im)
        ref_lp = butterworth(ref_im, high_pass=False, channel_axis=-1, npad=10, order=2)
        eps = np.finfo(float).eps
        im1 = np.multiply(base_im, (ref_im / (ref_lp + eps)))
        im1 = match_histograms(im1, base_im)
        sharpened_cube[:, :, i] = im1

    return sharpened_cube
