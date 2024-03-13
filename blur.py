import requests
import cv2
import numpy as np
import os
import xarray as xr
import netCDF4
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np
import scipy.interpolate as si
import scipy.optimize as so


class blurredCube:
    def __init__(self):
        self.image_height = None
        self.image_width = None
        self.bands = None
        self.cube = None
        self.array = None
        self.lsf = lsf()
        self.info = None
        self.folder_name = "C:/Users/Kim/Documents/master/HYPSO-1/hico_data"
        self.cubes = None
        self.path_to_np = ''
        self.sbi = 750

    def compare_fwhm(self, cube, desired_fwhm, sigma):
        # Calculate the actual FWHM
        actual_fwhm = self.blur_cube(cube, sigma)

        # Chech if desired fwhm is larger than actual fwhm
        if desired_fwhm < actual_fwhm:
            while abs(actual_fwhm - desired_fwhm) > 0.03:
                sigma += 0.1  # Increase sigma
                actual_fwhm = self.blur_cube(cube, sigma)
        elif desired_fwhm > actual_fwhm:
            while abs(actual_fwhm - desired_fwhm) > 0.03:
                sigma += 0.1 # Increase sigma
                actual_fwhm = self.blur_cube(cube, sigma)
        


        return cube

    def get_cube(self) -> np.ndarray:
        """Get the raw data from the folder.

        Returns:
            np.ndarray: The raw data.
        """
        # find file ending in .nc
        for file in os.listdir(self.folder_name["folder_name"]):
            if file.endswith(".np"):
                self.path_to_np = os.path.join(
                    self.info["folder_name"], file)
                break

        # Data from wavelengths less than 400 nm and greater than 900 nm are not recommended for analysis, but we will use them anyway, we can throw data away if needed, ask sivert

    
    def read_cube(self):
        print(self.cube)
        data = xr.open.dataarray(self.path_to_np) / 50.0
        # Access the variable that contains the band wavelengths
        band_wavelengths = data['wavelength']
        band_index = (np.abs(band_wavelengths - self.sbi)).argmin()

        image_data = data['data']
        center_wavelength_data = image_data.sel(band=band_index)


    def get_hico_images(self):
        # do something
        return self.cube
    
    def fwhm_calc(self):
        self.cube.get_fwhm_val()
            

    def blur_cube(self, sigma):
        """
            Blurs the cube in question

            args:
                The cube to be sharpened

            Returns:
                Blurred cube
        """
        for bands in self.cube:
            fwhm = self.lsf.get_fwhm_val(self.cube[bands])
            self.cube[bands] = cv2.GaussianBlur(self.cube[bands], (5, 5), sigma)


class hicodata:
    def __init__(self):
        files_folder = None
        data = None
        cube = []

    # Example from NASAs website 
    def get_hico_data_from_web(self):
        # Set the URL string to point to a specific data URL. Some generic examples are:
        #   https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/path/to/granule.nc4

        URL = 'https://oceandata.sci.gsfc.nasa.gov/directdataaccess/Level-1B/HICO/2014/042/'

        # Set the FILENAME string to the data file name, the LABEL keyword value, or any customized name. 
        filename = 'your_file_string_goes_here'
        result = requests.get(URL)
        try:
            result.raise_for_status()
            f = open(filename,'wb')
            f.write(result.content)
            f.close()
            print('contents of URL written to '+filename)
        except:
            print('requests.get() returned an error code '+str(result.status_code))


class lsf():
    def __init__(self):
        self.sigma_values = []
        self.bands = None
        self.center_wavelength = None
        self.hico_image_edge = None
        self.current_fwhm = []
        self.new_fwhm = []
        self.blurriest_fwhm = 3.5 
        self.sharpest_fwhm = 1.5

    def generate_desired_fwhm(self):
        if len(self.new_fwhm) == 0:
            self.parabole_func()
            print("Generated desired parabole FWHM curve")
        else:
            print("List is already generated.")


    def parabole_func(self):
        a_1 = -2/((self.bands/2)**2)
        for band_index in enumerate(self.bands):
            if band_index == 0:
                return self.new_fwhm.append(self.blurriest_fwhm)
            elif band_index < self.bands/2:
                self.new_fwhm.append(- (a_1) * (self.bands/2) ** 2 + self.sharpest_fwhm) # using the parabole function
            elif (band_index > self.bands/2 and band_index < len(self.bands)):
                a_2 = -((self.sharpest_fwhm-self.blurriest_fwhm)/((self.bands)/2)^2 - len(self.bands))
                b = self.blurriest_fwhm - a_2* ((self.bands)/2)**2
                self.new_fwhm.append((a_2) * (self.bands/2) ** 2 + b)

    def detect_sharpest_edge(image):
        """
            Detect the sharpest edge of the image.
        """
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gradient_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=5)
        gradient_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        max_gradient = np.max(gradient_magnitude)
        sharpest_edge = np.where(gradient_magnitude == max_gradient)

        return sharpest_edge

    def func(x, a, b, c, d):
        return a/(1+np.exp((x-b)/c)) + d

    def get_fwhm_val(self, image_edge):
        """ Get the FWHM value from the edge of the image.

        Parameters
        ----------
        image_edge : array
            The edge of the image.

        Returns
        -------
        fwhm : float
            The FWHM value.
        """

        N = 1000
        Np = len(image_edge)

        x0 = np.linspace(1.0, Np, Np)
        x0_interp = np.linspace(1.0, Np, N)

        image_edge_interp_1 = si.griddata(
            x0, image_edge, x0_interp, method='cubic')
        image_edge_interp_1_linear = si.griddata(
            x0, image_edge, x0_interp, method='linear')


        # f(x) = d + a / (1 + exp((x-b)/c))
        d = np.min(image_edge)
        b = Np//2+1  # closest integer larger than half the edge length?
        c = -0.5
        a = 2*(image_edge_interp_1[N//2] - d)

        #fit = so.curve_fit(func, x0, image_edge, p0=[a, b, c, d])
        try:
            fit = so.curve_fit(
                self.func, x0_interp, image_edge_interp_1_linear, p0=[a, b, c, d])
        except RuntimeError:
            return -1

        image_edge_interp_3 = self.func(
            x0_interp, fit[0][0], fit[0][1], fit[0][2], fit[0][3])


        image_edge_interp_3_normalized = (image_edge_interp_3-image_edge_interp_3.min())/(
            image_edge_interp_3.max()-image_edge_interp_3.min())

        image_line_interp = np.abs(np.diff(image_edge_interp_3_normalized))
        x1_interp = x0_interp[:-1]+(x0_interp[1]-x0_interp[0])/2

        half_max = image_line_interp.max()/2
        larger_than_indices = np.where(image_line_interp > half_max)[0]

        fwhm_0 = larger_than_indices[0]
        fwhm_1 = larger_than_indices[-1]
        #fwhm = (fwhm_1 - fwhm_0)*(Np-1.0)/N
        fwhm = x1_interp[fwhm_1]-x1_interp[fwhm_0]

        return fwhm        




        



        


