class datatype:
    def __init__(self):
        """Initialize the data class.

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

class blurredImage:
    def __init__(self) -> None:
        self.image_height = None
        self.image_width = None
        self.bands = None
        self.image = None
    
    def get_hico_images(self, )
        

