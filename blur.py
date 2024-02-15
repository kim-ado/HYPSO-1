import requests
import cv2
import numpy as np
import os
import xarray as xr
import netCDF4
import pandas as pd
import holoview as hv
from sklearn import svm
import sklearn
from sklearn.model_selection import GridSearchCV
import visacc
import maskacc
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si
import scipy.optimize as so
hv.notebook_extension('matplotlib')



class blurredCube:
    def __init__(self) -> None:
        self.image_height = None
        self.image_width = None
        self.bands = None
        self.cube = None
        self.array = None
    
    def get_hico_images(self):
        # do something

        return self.cube
    
    def get_cube(self) -> np.ndarray:
        """Get the raw data from the folder.

        Returns:
            np.ndarray: The raw data.
        """
        # find file ending in .nc
        for file in os.listdir(self.info["folder_name"]):
            if file.endswith(".np"):
                path_to_np = os.path.join(
                    self.info["folder_name"], file)
                break

        cube.values[cube.values<0] = np.nan

        cube = xr.open.dataarray(path_to_np)

        self.image_width = cube.

        # Data from wavelengths less than 400 nm and greater than 900 nm are not recommended for analysis, but we will use them anyway

        return cube
    
    def blur_cube(self, cube):
        """
            Blurs the cube in question

            args:
                The cube to be sharpened

            Returns:
                Blurred cube
        """
        for i in range(cube.shape[2]):
            exit ## 

class hicodata:
    def __init__(self):
        files_folder = None
        data = None

    def getHicoDataFromWeb(self):
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
        self.parabole_func = self._parabole_func
        self.bands = None
        self.center_wavelength = None
        self.hico_image_edge = None

    def _parabole_func(self, y):
        return np.power(4*y, 0.5)
    
    def segmentation_and_values_of_parabole(self, bands, center_wavelength):
        self.get_fwhm_val(self, image_edge)

    def generate_parabole_sigma_values(self, start_wavelength, center_wavelength, bands):
        """
            Helper function to generate wavelength dependent blur values for gaussian blur

            Returns:
                array for sigma values
        """
        for i in range((len(bands)-1)/2):
            sigma = i 
            self.sigma_values.append()
    
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




        



        


