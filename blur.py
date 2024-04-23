import cv2
import numpy as np
import os
import xarray as xr
import numpy as np
import scipy.interpolate as si
import scipy.optimize as so
import netCDF4 as nc
import matplotlib.pyplot as plt
from skimage.measure import profile_line



class blurCube():
    def __init__(self):
        self.sbi = 96 #Highest band index that is not trash
        self.mbi = 9 #Lowest band index that is not trash

        self.wavelengths = 0
        self.hico_image_edge = None

        self.current_fwhm = []
        self.desired_fwhm = []

        self.blurriest_fwhm = 3.5
        self.sharpest_fwhm = 1.5

        self.guessed_sigma = []
        self.sigma_values = [] # Temp storage for sigma values to use in the GaussianBlur function

        self.cube = []
        self.blurred_cube = []
        self.edge = None

        self.folder_name = "hico_data"

    def blur_cube(self):

        self.parabole_func()

        print("Desired FWHM: ", self.desired_fwhm)
        
        for i in range(self.mbi, self.sbi):
            
            lower = 0.01
            upper = 5.00
            epsilon = 0.01

            self.edge = self.convert_coordinates_to_intensity_values(self.cube.sel(bands=i).values, self.line)
            fwhm = self.get_fwhm_val(self.edge)
            print("Initial fwhm: ", fwhm)

            self.current_fwhm.append(fwhm)
            self.blurred_cube = xr.DataArray(
                data=np.zeros_like(self.cube.isel(bands=slice(self.mbi, self.sbi)).values),
                coords=self.cube.isel(bands=slice(self.mbi, self.sbi)).coords,
                dims=self.cube.isel(bands=slice(self.mbi, self.sbi)).dims
            )

            while upper - lower > epsilon:
                middle = (lower + upper) / 2
                self.blurred_cube.isel(bands=i-9).values = cv2.GaussianBlur(self.cube.sel(bands=i).values, (0,0), sigmaX=middle)

                self.edge = self.convert_coordinates_to_intensity_values(self.blurred_cube.isel(bands=i-9).values, self.line)
                fwhm = self.get_fwhm_val(self.edge)
                
                self.current_fwhm[i-self.mbi] = fwhm
                print("current fwhm: ", self.current_fwhm[i-self.mbi])

                if self.current_fwhm[i-self.mbi] > self.desired_fwhm[i-self.mbi]:
                    upper = middle
                else:
                    lower = middle

            final_sigma = (lower + upper) / 2
            self.sigma_values.append(final_sigma)
            self.blurred_cube.isel(bands=i-self.mbi).values = cv2.GaussianBlur(self.cube.sel(bands=i).values, (0,0), sigmaX=final_sigma[i])
            self.blurred_cube.isel(bands=-i-self.mbi).values = cv2.GaussianBlur(self.cube.sel(bands=-i).values, (0,0), sigmaX=final_sigma[i])
    
    def get_cube(self):
        """Get the raw data from the folder.

        Returns:
            np.ndarray: The raw data.
        """
        # find file ending in .nc
        for file in os.listdir(self.folder_name):
            if file.endswith(".nc"):
                self.path_to_nc = os.path.join(
                    self.folder_name, file)
                print("File accessed: ", self.path_to_nc)
                break

        # Data from wavelengths less than 400 nm and greater than 900 nm are not recommended for analysis, but we will use them anyway, we can throw data away if needed, ask sivert

    
    def read_cube(self):
        #f = nc.Dataset(self.path_to_nc, 'r')
        ds = xr.open_dataset(self.path_to_nc, group='products', engine='h5netcdf')
        Lt = ds['Lt']
        self.wavelengths = Lt.attrs['wavelengths']

        slope = 0.02  # The slope value mentioned in the documentation
        Lt_corrected = Lt * slope

        self.bands = len(self.wavelengths[self.mbi:self.sbi])

        print("Bands: ", self.bands)
        
        self.cube = Lt_corrected


    def detect_sharpest_edge(self, image):
        """
            Detect the sharpest edge of the image.
        """
        image = self.cube.sel(bands=96).values
        image = image - np.min(image)  # Shift the range so that it starts from 0
        image = image / np.max(image)  # Normalize to the range [0, 1]
        image = (image * 255).astype(np.uint8)  # Scale to the range [0, 255] and convert to 8-bit integers

        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=10, maxLineGap=250)
        
        if lines is None:
            return None, None
        
        long_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length > 8 and y1 == y2:
                long_lines.append(line)

        # Replace 'lines' with the filtered list
        lines = long_lines

        for line in lines:
            print("line:", line)
        

    def visualize_cube(self):
        # Assuming 'cube' is your xarray Dataset
        R = self.cube.sel(bands=42).values
        G = self.cube.sel(bands=27).values
        B = self.cube.sel(bands=11).values

        # Stack the R, G, B bands to create a 3D array (image)
        rgb_image = np.dstack((R, G, B))
        
        image = self.cube.sel(bands=96).values
        image = image - np.min(image)  # Shift the range so that it starts from 0
        image = image / np.max(image)  # Normalize to the range [0, 1]
        image = (image * 255).astype(np.uint8)  # Scale to the range [0, 255] and convert to 8-bit integers

        # Normalize the image to the range [0, 1] because matplotlib expects values in this range
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

        edges = cv2.Canny(blurred_image, 50, 150, apertureSize=3)

        # Detect lines using Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=6, maxLineGap=20)

        # Filter the lines
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if y1 == y2 and 15 <= length <= 30:  # Horizontal line with length between 15 and 30
                horizontal_lines.append(line)

        # Get the line at index 0
        line = horizontal_lines[0][0]
        print("line:", line)

        # Saving the coordinates of the line
        self.line = line

        # Get the pixel intensity values along the line
        self.edge = self.convert_coordinates_to_intensity_values(self.cube.sel(bands=96).values, self.line)

        print("Edge:", self.edge)
        """
        line_image = np.zeros_like(image)

        rgb_image_with_lines = np.copy(rgb_image)

        for line in horizontal_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

        for line in horizontal_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(rgb_image_with_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)


        plt.subplot(1, 3, 1)
        plt.imshow(rgb_image)
        plt.title('Original RGB Image')

        # Display the image with the lines
        plt.subplot(1, 3, 2)
        plt.imshow(line_image, cmap='gray')
        plt.title('Image with Lines')

        # Display the RGB image with the lines
        plt.subplot(1, 3, 3)
        plt.imshow(rgb_image_with_lines)
        plt.title('RGB Image with Lines')

        plt.show()

        """
    def parabole_func(self): # Fra index 9 til index 95 ettersom at det er litt dårlig
        bands = self.bands
        a_1 = -2/((bands/2)**2)
        for band in range(bands):  # Iterate over the range of bands
            if band == 0:
                self.desired_fwhm.append(self.blurriest_fwhm)
            elif band < bands/2:
                self.desired_fwhm.append(-a_1 * (band - bands/2) ** 2 + self.sharpest_fwhm) # using the parabole function
            elif (band >= bands/2 and band < bands):
                self.desired_fwhm.append(-a_1 * ((bands - band) - bands/2) ** 2 + self.sharpest_fwhm)

    def convert_coordinates_to_intensity_values(self, image, line):
        """Convert the edge to intensity values.

        Parameters
        ----------
        image : 2D array
            The image.
        point1, point2 : tuple
            The endpoints of the line.

        Returns
        -------
        image_edge : array
            The intensity values of the edge.
        """

        # Get the endpoints of the line
        point1 = (line[1], line[0])  # (y1, x1)
        point2 = (line[3], line[2])  # (y2, x2)

        # Extract the pixel intensity values along the line
        image_edge = profile_line(image, point1, point2)

        return image_edge

    
    @staticmethod
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




        



        


