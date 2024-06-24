import numpy as np
import spectral.io.envi as envi
import metric
import h1data


class sharp:
    def __init__(self):
        self.initial_cube = None
        self.refrence_cube = None

        self.sharpened_cube = None
        self.sharpest_band_index = None

        # metrics - full refrence
        self.sam = {}
        self.ergas = {}
        self.uiqi = {}
        
        # metrics - no reference
        self.d_s = {}
        self.d_rho = {}
        self.d_lambda = {}
        self.qnr = {}
        
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
    
    def ranking_bands(self):
        """ Helper function to find which bands are the same sharpness

        Args: 
            Cube to 
        """


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

    # Standardize the data
    mean_image = np.mean(image, axis=(0, 1))
    std_image = np.std(image, axis=(0, 1))
    standardized_image = (image - mean_image) / std_image

    # Do PCA on standardized image
    img_variable_form = np.reshape(
        standardized_image, (standardized_image.shape[0] * standardized_image.shape[1], standardized_image.shape[2]))
    
    U, S, Vh = svd(img_variable_form, full_matrices=False)
    principal_components = np.dot(U, np.diag(S))
    component_cube = np.reshape(
        principal_components, (standardized_image.shape[0], standardized_image.shape[1], standardized_image.shape[2]))

    # Standardize the sharpest band
    sharpest_band = (image[:, :, sharpest_band_index] - mean_image[sharpest_band_index]) / std_image[sharpest_band_index]

    # Match histogram of standardized sharpest band to first component
    matched_sharpest_band = match_histograms(
        sharpest_band, component_cube[:, :, 0])

    # Replace first component with matched sharpest band
    fixed_component_cube = np.copy(component_cube)
    fixed_component_cube[:, :, 0] = matched_sharpest_band
    fixed_cc_variable_form = np.reshape(
        fixed_component_cube, (standardized_image.shape[0] * standardized_image.shape[1], standardized_image.shape[2]))

    # Do inverse PCA
    sharpend_variable_form = np.dot(fixed_cc_variable_form, Vh)

    # Reverse the standardization
    sharpend_variable_form = sharpend_variable_form * std_image + mean_image

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

    # Ensure the input images are float32
    base_image = base_image.astype(np.float32)
    reference_image = reference_image.astype(np.float32)
    
    # Perform wavelet transform on base and reference images
    base_coeffs = pywt.wavedec2(base_image, wavelet, level=level)
    reference_coeffs = pywt.wavedec2(reference_image, wavelet, level=level)

    # Replace detail coefficients with those from the reference image
    sharpened_coeffs = list(base_coeffs)
    for i in range(1, len(sharpened_coeffs)):
        sharpened_coeffs[i] = reference_coeffs[i]

    # Perform inverse wavelet transform and ensure output is float32
    sharpened_image = pywt.waverec2(sharpened_coeffs, wavelet).astype(np.float32)

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
                "ladder",
                "ladder_bottom"
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

    def string(self):
        if self.type == "wavelet":
            return f"{self.type}_{self.mother_wavelet}_{self.wavelet_level}_{self.strategy}"
        elif self.type == "laplacian":
            return f"{self.type}_{self.filter_order}_{self.strategy}"
        elif self.type == "cs":
            return f"{self.type}"
        elif self.type == "none":
            return f"{self.type}"
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
            sharpened_cube = wavelet_cube_sharpen_stepwise(
                cube, sbi, self.mother_wavelet, self.wavelet_level)
        elif self.strategy == "ladder_bottom":
            sharpened_cube = wavelet_cube_sharpen_stepwise_next_order(
                cube, sbi, self.mother_wavelet, self.wavelet_level)
        return sharpened_cube

    def laplacian_cube_sharpen(self, cube: np.ndarray, sbi: int) -> np.ndarray:
        sharpened_cube = np.zeros(cube.shape)
        if self.strategy == "regular":
            sharpened_cube = laplacian_cube_sharpen_regular(
                cube, sbi, self.filter_order)
        if self.strategy == "ladder":
            sharpened_cube = laplacian_cube_sharpen_stepwise(
                cube, sbi, self.filter_order)
        if self.strategy == "ladder_bottom":
            sharpened_cube = laplacian_cube_sharpen_stepwise_next_order(
                cube, sbi, self.filter_order)
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



def wavelet_cube_sharpen_stepwise(cube: np.ndarray, sbi: int, mother_wavelet: str, wavelet_level: int) -> np.ndarray:
    """Sharpen a cube using the wavelet sharpening algorithm with a stepwise strategy.

    Args:
        cube (np.ndarray): The cube to sharpen.
        sbi (int): The Sharpest Base Image Index in the cube.
        mother_wavelet (str): The mother wavelet to use.
        wavelet_level (int): The level of the wavelet transform.

    Returns:
        np.ndarray: The sharpened cube.
    """
    sharpened_cube = np.zeros(cube.shape)
    band_order = determine_band_order(cube, sbi)

    for i in range(cube.shape[2]):
        base_index = band_order[i]
        ref_index = sbi

        sharpened_cube[:, :, base_index] = wavelet_sharpen(
            cube[:, :, base_index], cube[:, :, ref_index], mother_wavelet, wavelet_level)

    return sharpened_cube

def wavelet_cube_sharpen_stepwise_next_order(cube: np.ndarray, sbi: int, mother_wavelet: str, wavelet_level: int) -> np.ndarray:
    """Sharpen a cube using the wavelet sharpening algorithm with a stepwise strategy.

    Args:
        cube (np.ndarray): The cube to sharpen.
        sbi (int): The Sharpest Base Image Index in the cube.
        mother_wavelet (str): The mother wavelet to use.
        wavelet_level (int): The level of the wavelet transform.

    Returns:
        np.ndarray: The sharpened cube.
    """
    
    sharpened_cube = np.zeros(cube.shape)
    
    for i in range(sbi):
        base_index = i
        ref_index = i + 1
        sharpened_cube[:, :, base_index] = wavelet_sharpen(
            cube[:, :, base_index], cube[:, :, ref_index], mother_wavelet, wavelet_level)
        
        for j in range(i):
            base_index_left = j
            ref_index_left = ref_index
            sharpened_cube[:, :, base_index_left] = wavelet_sharpen(
                cube[:, :, base_index_left], cube[:, :, ref_index_left], mother_wavelet, wavelet_level)

    for i in range(cube.shape[2] - 2, sbi - 1, -1):
        base_index = i + 1 
        ref_index = i 
        sharpened_cube[:, :, base_index] = wavelet_sharpen(
            cube[:, :, base_index], cube[:, :, ref_index], mother_wavelet, wavelet_level)
        
        for j in range(cube.shape[2] - 2, i, -1):
            base_index_right = j + 1 
            ref_index_right = ref_index 
            sharpened_cube[:, :, base_index_right] = wavelet_sharpen(
                cube[:, :, base_index_right], cube[:, :, ref_index_right], mother_wavelet, wavelet_level)

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

def laplacian_cube_sharpen_stepwise(cube: np.ndarray, sbi: int, filter_order: int = 5) -> np.ndarray:
    """Sharpen a cube using the Laplacian sharpening algorithm with a ladder strategy

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
    band_order = determine_band_order(cube, sbi)

    for i in range(cube.shape[2]):
        base_index = band_order[i]
        ref_im = cube[:, :, sbi]
        base_im = cube[:, :, i]
        ref_im = np.multiply((ref_im - np.mean(ref_im)) , (np.std(base_im) / np.std(ref_im))) + np.mean(base_im)
        ref_lp = butterworth(ref_im, high_pass=False, channel_axis=-1, npad=10, order=2)
        eps = np.finfo(float).eps
        im1 = np.multiply(base_im, (ref_im / (ref_lp + eps)))
        im1 = match_histograms(im1, base_im)
        sharpened_cube[:, :, i] = im1

    return sharpened_cube


def laplacian_cube_sharpen_stepwise_next_order(cube: np.ndarray, sbi: int, filter_order: int = 5) -> np.ndarray:
    """Sharpen a cube using the Laplacian sharpening algorithm ladder bottom up strategy

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
    
    for i in range(sbi):
        base_index = i
        ref_index = i + 1
        sharpened_cube[:, :, base_index] = sharpen_band_laplacian(cube, ref_index, base_index)
        
        for j in range(i):
            base_index_left = j
            ref_index_left = ref_index
            sharpened_cube[:, :, base_index_left] = sharpen_band_laplacian(cube, ref_index_left, base_index_left)

    for i in range(cube.shape[2] - 2, sbi - 1, -1):
        base_index = i + 1 
        ref_index = i 
        sharpened_cube[:, :, base_index] = sharpen_band_laplacian(cube, ref_index, base_index)
        
        for j in range(cube.shape[2] - 2, i, -1):
            base_index_right = j + 1 
            ref_index_right = ref_index 
            sharpened_cube[:, :, base_index_right] = sharpen_band_laplacian(cube, ref_index_right, base_index_right)
    
    return sharpened_cube

def sharpen_band_laplacian(cube, ref_index, base_index):
    import numpy as np
    from skimage.filters import butterworth
    from skimage.exposure import match_histograms
    
    ref_im = cube[:, :, ref_index]
    base_im = cube[:, :, base_index]
    ref_im = np.multiply((ref_im - np.mean(ref_im)), (np.std(base_im) / np.std(ref_im))) + np.mean(base_im)
    ref_lp = butterworth(ref_im, high_pass=False, channel_axis=-1, npad=10, order=2)
    eps = np.finfo(float).eps
    im1 = np.multiply(base_im, (ref_im / (ref_lp + eps)))
    im1 = match_histograms(im1, base_im)
    return im1


def determine_band_order(cube: np.ndarray, sbi: int) -> np.ndarray:
    """Determine the order of bands for ladder approach

    Args:
        cube (np.ndarray): The cube.
        sbi (int): The Sharpest Base Image Index in the cube.

    Returns:
        np.ndarray: The order of bands 
    """
    distances = np.abs(np.arange(cube.shape[2]) - sbi) % cube.shape[2]

    band_order = np.argsort(distances)
    
    return band_order



