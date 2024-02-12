import requests

class hicodata:
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

    def getHicoDataFromWeb(self):
        # Set the URL string to point to a specific data URL. Some generic examples are:
        #   https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/path/to/granule.nc4

        URL = 'your_URL_string_goes_here'

        # Set the FILENAME string to the data file name, the LABEL keyword value, or any customized name. 
        FILENAME = 'your_file_string_goes_here'

        import requests
        result = requests.get(URL)
        try:
            result.raise_for_status()
            f = open(FILENAME,'wb')
            f.write(result.content)
            f.close()
            print('contents of URL written to '+FILENAME)
        except:
            print('requests.get() returned an error code '+str(result.status_code))


class blurredImage:
    def __init__(self) -> None:
        self.image_height = None
        self.image_width = None
        self.bands = None
        self.image = None
    
    def get_hico_images(self):
        self.raw_image = hicodata
        image = hicodata.getdata
        


