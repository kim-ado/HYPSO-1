import plotly.express as px
import os

import numpy as np
# import spectral.io.envi as envi # add support for reading envi files later


class h1data:
    def __init__(self, top_folder_name):
        """Initialize the h1data class.

        Args:
            top_folder_name (str): The name of the top folder of the data.

        Raises:
            ValueError: If no folder with substring 'hsi0' is found.


        """
        # Get the metadata to be used in the class
        self.info = self.get_metainfo(top_folder_name)
        self.raw_cube = self.get_raw_cube()  # consider initializing with None

        # Set the radiometric and spectral coefficients
        self.rad_file = "rad_coeffs_FM_binx9_2022_08_06_Finnmark_recal_a.csv"
        self.radiometric_coefficients = self.set_radiometric_coefficients(
            path=None)

        # Set the spectral coefficients
        self.spec_file = "spectral_bands_HYPSO-1_120bands.csv"
        self.spec_coefficients = self.set_spectral_coefficients(path=None)

        # Calibrate the raw data
        self.wavelengths = self.spec_coefficients
        self.l1a_cube = self.calibrate_cube()  # consider initializing with None

        # Set the center wavelength
        # found empirically using the brisque metric
        self.center_wavelength = np.argmin(
            np.abs(self.spec_coefficients-553))

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
            raise ValueError("Could not read spectral coefficients file.")

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

        # reverse the order of the third dimension
        cube = cube[:, :, ::-1]

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
        # self.wavelengths = self.spec_coefficients

        return cube_calibrated

    def show_raw_cube(self) -> None:
        """Show the raw data cube."""
        obj = px.imshow(np.rot90(self.raw_cube[:, :, 47]), aspect=4.5)
        obj.show()

    def show_l1a_cube(self) -> None:
        """Show the l1a cube."""
        obj = px.imshow(np.rot90(self.l1a_cube[:, :, 47]), aspect=4.5)
        obj.show()

    def write_geojson(self) -> None:
        import shutil
        import json
        """Write the geojson metadata file.
        
        Args:
            writingmode (str, optional): The writing mode. Defaults to "w".

        Raises:
            ValueError: If the position file could not be found.            
        """

        # check if folder exists
        metadatapath = "geojsonimgs"
        if not os.path.exists(metadatapath):
            os.mkdir(metadatapath)

        # convert dictionary to json
        geojsondict = {}

        geojsondict["type"] = "Feature"

        pos_file = ""
        foldername = os.path.join("data", self.info["folder_name"])
        for file in os.listdir(foldername):
            if file.endswith("geometric-meta-info.txt"):
                pos_file = os.path.join(foldername, file)
                break

        if pos_file == "":
            raise ValueError(f"Could not find position file in {foldername}")

        with open(pos_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "lat lon" in line:
                    lat = float(line.split("lat lon")[1].split(" ")[1])
                    lon = float(line.split("lat lon")[1].split(" ")[2])
                    break

        geojsondict["geometry"] = {}
        geojsondict["geometry"]["type"] = "Point"
        geojsondict["geometry"]["coordinates"] = [lon, lat]

        geojsondict["properties"] = {}
        name = self.info["folder_name"].split("CaptureDL_")[-1].split("_")[0]
        geojsondict["properties"]["name"] = name

        geojsondict["metadata"] = {}
        date = self.info["folder_name"].split(
            "CaptureDL_")[-1].split("20")[1].split("T")[0]
        date = f"20{date}"

        time = self.info["folder_name"].split(
            "CaptureDL_")[-1].split("20")[1].split("T")[-1]

        geojsondict["metadata"]["date"] = date
        geojsondict["metadata"]["time"] = time
        geojsondict["metadata"]["frames"] = self.info["frame_count"]
        geojsondict["metadata"]["lines"] = self.info["image_height"]
        geojsondict["metadata"]["bands"] = self.info["image_width"]
        geojsondict["metadata"]["satellite"] = "HYPSO-1"

        img_file = ""
        foldername = os.path.join("data", self.info["folder_name"])
        for file in os.listdir(foldername):
            if file.endswith("bin3.png"):
                # get path to png
                img_file = os.path.join(foldername, file)
                break

        # copy img_file to metadatapath
        dst_png = f"{name}_{date}_{time}.png"
        dst_json = f"{name}_{date}_{time}.geojson"
        shutil.copy(img_file, os.path.join(
            metadatapath, dst_png))

        geojsondict["properties"]["datalink"] = os.path.join(
            metadatapath, dst_png)

        dst_json = os.path.join(metadatapath, dst_json)
        with open(dst_json, "w") as f:
            json.dump(geojsondict, f, indent=4)

    # def write_netcdf4file(self, dst: str = ""):
    #     """Write the netcdf4 file.

    #     Args:
    #         dst (str, optional): The destination. Defaults to "".
    #     """
    #     import xarray as xr

    #     if dst == "":
    #         dst = os.path.join("data", self.info["folder_name"], "l1a.nc")

    #     # create dataset
    #     ds = xr.Dataset(
    #         {
    #             "l1a_cube": (["frame", "line", "band"], self.l1a_cube),
    #             "wavelengths": (["band"], self.wavelengths)
    #         },
    #         coords={
    #             "frame": np.arange(self.info["frame_count"]),
    #             "line": np.arange(self.info["image_height"]),
    #             "band": np.arange(self.info["image_width"]),
    #         },
    #     )

    #     # write to file
    #     ds.to_netcdf(dst)


def get_rgb_matrix(cube: h1data, equalized: bool = False) -> np.ndarray:
    """Get the RGB matrix from the cube.

    Args:
        cube (h1data): The cube.
        gamma (float, optional): The gamma value. Defaults to 0.

    Returns:
        np.ndarray: The RGB matrix.
    """
    from skimage import exposure

    wl = cube.wavelengths
    R_ind = np.argmin(abs(wl-600))
    G_ind = np.argmin(abs(wl-553))
    B_ind = np.argmin(abs(wl-500))

    rgb_matrix = cube.l1a_cube[:, :, [R_ind, G_ind, B_ind]]

    if equalized:
        def normalize(arr):
            arr = arr - np.min(arr)
            return arr / np.max(arr)

        rgb_matrix = exposure.equalize_adapthist(normalize(rgb_matrix))

    return rgb_matrix
