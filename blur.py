import cv2
import numpy as np
import os
import xarray as xr
import numpy as np
import scipy.interpolate as si
import scipy.optimize as so
import netCDF4 as nc            


class blurCube():
    def __init__(self):
        self.sbi = 753
        self.hico_image_edge = None

        self.current_fwhm = []
        self.desired_fwhm = []

        self.blurriest_fwhm = 3.5
        self.sharpest_fwhm = 1.5

        self.guessed_sigma = []
        self.sigma_values = [] # Temp storage for sigma values to use in the GaussianBlur function

        self.cube = None
        self.blurred_cube = None

        self.folder_name = "hico_data"

    def blur_cube(self, cube):

        self.desired_fwhm = self.generate_desired_fwhm()

        try:
            if not self.edge:
                self.edge = self.detect_sharpest_edge(cube[self.sbi]) # Finding the sharpest edge of the image at the center wavelength
        except Exception as e:
            print("Error occurred while detecting the sharpest edge:", str(e))

        for i in range(cube.shape[2]/2):
            
            lower = 0.01
            upper = 5.00
            epsilon = 0.01

            fwhm = self.get_fwhm_val(self.cube[i], self.edge)
            self.current_fwhm.append(fwhm)

            while upper - lower > epsilon:
                middle = (lower + upper) / 2
                self.blurred_cube[i] = cv2.GaussianBlur(self.cube[i], (5,5), sigma=middle)
                fwhm = self.get_fwhm_val(self.blurred_cube[i], self.edge)
                self.current_fwhm[i] = fwhm

                if self.current_fwhm[i] > self.desired_fwhm[i]:
                    upper = middle
                else:
                    lower = middle

            final_sigma = (lower + upper) / 2
            self.sigma_values.append(final_sigma)
            self.blurred_cube[i] = cv2.GaussianBlur(self.cube[i], (5,5), sigma=final_sigma[i])
            self.blurred_cube[-i] = cv2.GaussianBlur(self.cube[-i], (5,5), sigma=final_sigma[i])
        
    
    def get_cube(self):
        """Get the raw data from the folder.

        Returns:
            np.ndarray: The raw data.
        """
        # find file ending in .nc
        for file in os.listdir(self.folder_name):
            if file.endswith("L1B_ISS"):
                self.path_to_nc = os.path.join(
                    self.folder_name, file)
                print(self.path_to_nc)
                break

        # Data from wavelengths less than 400 nm and greater than 900 nm are not recommended for analysis, but we will use them anyway, we can throw data away if needed, ask sivert

    
    def read_cube(self):
        print(self.cube)
        self.cube = xr.open_dataset(self.path_to_nc, engine='netcdf4') / 50.0
        f = nc.Dataset(self.path_to_nc, 'r')
        print(f)
        # Access the variable that contains the band wavelengths
        print(self.cube)

    def generate_desired_fwhm(self):
        if len(self.desired_fwhm) == 0:
            self.parabole_func()
            print("Generated desired parabole FWHM curve")
        else:
            print("List is already generated.")


    def parabole_func(self):
        bands = len(self.cube[2])
        a_1 = -2/((bands/2)**2)
        for band_index in enumerate(bands):
            if band_index == 0:
                return self.desired_fwhm.append(self.blurriest_fwhm)
            elif band_index < bands/2:
                self.desired_fwhm.append(- (a_1) * (bands/2) ** 2 + self.sharpest_fwhm) # using the parabole function
            elif (band_index > bands/2 and band_index < len(bands)):
                a_2 = -((self.sharpest_fwhm-self.blurriest_fwhm)/((bands)/2)^2 - len(bands))
                b = self.blurriest_fwhm - a_2* ((bands)/2)**2
                self.desired_fwhm.append((a_2) * (bands/2) ** 2 + b)

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




        



        


