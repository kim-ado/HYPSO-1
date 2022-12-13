import os

import matplotlib.pyplot as plt
import numpy as np
import spectral.io.envi as envi


class h1data:
    def __init__(self, top_folder_name):
        """Initialize the h1data class.

        Args:
            top_folder_name (str): The name of the top folder of the data.

        Raises:
            ValueError: If no folder with substring 'hsi0' is found.


        """
        # get the metadata to be used in the class
        self.info = self.get_metainfo(top_folder_name)

        # set the radiometric and spectral coefficients
        self.rad_file = "rad_coeffs_FM_binx9_2022_08_06_Finnmark_recal_a.csv"
        self.radiometric_coefficients = self.set_radiometric_coefficients(
            path=None)

        # set the spectral coefficients
        self.spec_file = "spectral_bands_HYPSO-1_120bands.csv"
        self.spec_coefficients = self.set_spectral_coefficients(path=None)

        # Load the raw data
        self.raw_cube = self.get_raw_cube()  # consider initializing with None
        self.l1a_cube = self.calibrate_cube()  # consider initializing with None
        self.wavelengths = None

    def get_metainfo(self, top_folder_name: str) -> dict:
        """Get the metadata from the top folder of the data.

        Args:
            top_folder_name (str): The name of the top folder of the data.

        Returns:
            dict: The metadata.
        """
        info = {}
        info["top_folder_name"] = top_folder_name
        info["folder_name"] = top_folder_name.split("/")[-1]

        # find folder with substring "hsi0" or throw error
        for folder in os.listdir(top_folder_name):
            if "hsi0" in folder:
                raw_folder = folder
                break
        else:
            raise ValueError("No folder with metadata found.")

        # combine top_folder_name and raw_folder to get the path to the raw data
        config_file_path = os.path.join(
            top_folder_name, raw_folder, "capture_config.ini")

        def is_integer_num(n) -> bool:
            if isinstance(n, int):
                return True
            if isinstance(n, float):
                return n.is_integer()
            return False

        # read all lines in the config file
        with open(config_file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                # split the line at the equal sign
                line = line.split("=")
                # if the line has two elements, add the key and value to the info dict
                if len(line) == 2:
                    key = line[0].strip()
                    value = line[1].strip()
                    try:
                        if is_integer_num(float(value)):
                            info[key] = int(value)
                        else:
                            info[key] = float(value)
                    except:
                        info[key] = value

        info["background_value"] = 8*info["bin_factor"]

        info["x_start"] = info["aoi_x"]
        info["x_stop"] = info["aoi_x"] + info["column_count"]
        info["y_start"] = info["aoi_y"]
        info["y_stop"] = info["aoi_y"] + info["row_count"]
        info["exp"] = info["exposure"]/1000  # in seconds

        info["image_height"] = info["row_count"]
        info["image_width"] = int(info["column_count"] / info["bin_factor"])
        info["im_size"] = info["image_height"]*info["image_width"]

        return info

    def get_path_to_coefficients(self) -> str:
        """Get the path to the coefficients folder."""
        file_path = os.path.dirname(os.path.abspath(__file__))
        coeff_path = os.path.join(
            file_path, "cal-char-corr", "FM-calibration", "Coefficients")
        return coeff_path

    def set_spectral_coefficients(self, path: str) -> np.ndarray:
        """Set the spectral coefficients from the csv file.

        Args:
            path (str, optional): Path to the spectral coefficients csv file. Defaults to None.
            sets the spec_file attribute to the path.
            if no path is given, the spec_file path is used.


        Returns:
            np.ndarray: The spectral coefficients.
        """

        if path is None:
            coeff_path = self.get_path_to_coefficients()
            spectral_coeff_csv_name = self.spec_file
            spectral_coeff_file = os.path.join(
                coeff_path, spectral_coeff_csv_name)
        else:
            spectral_coeff_file = path
            self.spec_file = path
        try:
            spectral_coeffs = np.genfromtxt(
                spectral_coeff_file, delimiter=',')
        except:
            spectral_coeffs = None

        self.spec_coefficients = spectral_coeffs

        return spectral_coeffs

    def set_radiometric_coefficients(self, path: str) -> np.ndarray:
        """Set the radiometric coefficients from the csv file.

        Args:
            path (str, optional): Path to the radiometric coefficients csv file. Defaults to None.
            sets the rad_file attribute to the path.
            if no path is given, the rad_file path is used.

        Returns:
            np.ndarray: The radiometric coefficients.

        """
        if path is None:
            coeff_path = self.get_path_to_coefficients()
            radiometric_coeff_csv_name = self.rad_file
            radiometric_coeff_file = os.path.join(
                coeff_path, radiometric_coeff_csv_name)
        else:
            radiometric_coeff_file = path
            self.rad_file = path

        try:
            radiometric_coeffs = np.genfromtxt(
                radiometric_coeff_file, delimiter=',')
        except:
            radiometric_coeffs = None

        self.radiometric_coefficients = radiometric_coeffs
        self.calibrate_cube()
        return radiometric_coeffs

    def get_raw_cube(self) -> np.ndarray:
        """Get the raw data from the top folder of the data.

        Returns:
            np.ndarray: The raw data.
        """
        # find file ending in .bip
        for file in os.listdir(self.info["top_folder_name"]):
            if file.endswith(".bip"):
                path_to_bip = os.path.join(
                    self.info["top_folder_name"], file)
                break

        cube = np.fromfile(path_to_bip, dtype='uint16')
        cube = cube.reshape((-1, 684, 120))

        return cube

    def apply_radiometric_calibration(self, frame, exp, background_value, radiometric_calibration_coefficients):
        ''' Assumes input is 12-bit values, and that the radiometric calibration
        coefficients are the same size as the input image.

        Note: radiometric calibration coefficients have original size (684,1080),
        matching the "normal" AOI of the HYPSO-1 data (with no binning).'''

        frame = frame - background_value
        frame_calibrated = frame * radiometric_calibration_coefficients / exp

        return frame_calibrated

    def calibrate_cube(self) -> np.ndarray:
        """Calibrate the raw data cube."""

        background_value = self.info['background_value']
        x_start = self.info['x_start']
        x_stop = self.info['x_stop']
        exp = self.info['exp']
        image_height = self.info['image_height']
        image_width = self.info['image_width']

        # Radiometric calibration
        num_frames = self.info["frame_count"]
        cube_calibrated = np.zeros([num_frames, image_height, image_width])
        for i in range(num_frames):
            frame = self.raw_cube[i, :, :]
            frame_calibrated = self.apply_radiometric_calibration(
                frame, exp, background_value, self.radiometric_coefficients)
            cube_calibrated[i, :, :] = frame_calibrated

        self.l1a_cube = cube_calibrated
        self.wavelengths = self.spec_coefficients

        return cube_calibrated

    def show_raw_cube(self) -> None:
        """Show the raw data cube."""
        plt.imshow(np.rot90(self.raw_cube[:, ::10, 20]), aspect=3)

    def show_l1a_cube(self) -> None:
        """Show the l1a cube."""
        plt.imshow(np.rot90(self.l1a_cube[:, ::10, 20]), aspect=3)


class sharp:
    def __init__(self):
        self.initial_cube = None
        self.refrence_cube = None

        self.sharpened_cube = None
        self.sharpest_band_index = None

        # metrics - no refrence
        self.brisque = None

        # metrics - full refrence
        self.sam = None

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
            sharp_obj.initial_cube)[3] // 2

        sharp_obj.evaluate()

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
        sharp_obj.get_no_refrence_metric()

        return sharp_obj

    def get_brisque(self) -> np.ndarray:
        """Get the brisque metric.

        Returns:
            float: The brisque metric.
        """
        self.brisque = brisque(self.initial_cube)
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

        if self.initial_cube == None:
            raise ValueError("No initial cube loaded")
        elif self.refrence_cube == None:
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


def brisque(image: np.ndarray) -> np.ndarray:
    """Calculate the brisque metric.

    Args:
        image (np.ndarray): The image to be evaluated.

    Returns:
        np.ndarray: The brisque metric for each wavelength.
    """
    brisque_metric = np.zeros(image.shape[2])
    for i in range(image.shape[2]):
        # TODO: add brisque score for each wavelength
        brisque_metric[i] = np.mean((image[:, :, i]))

    return brisque_metric


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
