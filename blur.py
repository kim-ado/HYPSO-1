import cv2
import numpy as np
import os
import xarray as xr
import numpy as np
import scipy.interpolate as si
import scipy.optimize as so
import matplotlib.pyplot as plt
from skimage.measure import profile_line

class blurCube():
    def __init__(self):
        self.sbi = 96 #Highest band index that is not trash
        self.mbi = 9 #Lowest band index that is not trash

        # Sharpest band is the middle band which should be 44 serves as PAN image

        self.wavelengths = 0

        self.current_fwhm = []
        self.desired_fwhm = []

        self.blurriest_fwhm = 3.5
        self.sharpest_fwhm = 1.5

        self.sigma_values = [] # Final sigma values

        self.final_fwhm = []
        self.initial_fwhm = []

        self.cube = []
        self.blurred_cube = []
        self.edge = None

        self.folder_name = "hico_data/downloaded_data"
        self.patch_size = 170

    def blur_cubes(self):

        self.blur_cube()

        self.get_cubes()

        for paths in self.paths_to_nc:
            self.path_to_nc = paths
            self.read_cube()
            self.blurred_cube = xr.DataArray(
                data=np.zeros_like(self.cube.isel(bands=slice(self.mbi, self.sbi)).values),
                coords=self.cube.isel(bands=slice(self.mbi, self.sbi)).coords,
                dims=self.cube.isel(bands=slice(self.mbi, self.sbi)).dims
            )

            for i in range(self.mbi, self.sbi):
                self.blurred_cube.loc[dict(bands=i-self.mbi)] = cv2.GaussianBlur(self.cube.sel(bands=i).values, (0,0), sigmaX=self.sigma_values[i-self.mbi])


    def divide_into_patches(self, cube, bandindex):
        """
            Divide the blurred cube into patches of size patch_size.
        """
        from scipy.io import savemat

        ref = self.cube.sel(bands=bandindex).values

        pan = cube.sel(bands=44).values

        count = 1
        for i in range(0, cube.shape[1], self.patch_size):
            for j in range(0, cube.shape[2], self.patch_size):
                patch = cube[:, i:i+self.patch_size, j:j+self.patch_size]
                
                # Create a folder for each patch
                folder_name = f'hico_{str(count).zfill(2)}'
                file_name = os.path.join('hico', folder_name)
                if not os.path.exists(file_name):
                    os.makedirs(file_name)
                
                savemat(os.path.join(file_name, f'pavia_{str(count).zfill(2)}.mat'), {'ref': ref, 'downsampled': downsampled, 'pan': pan})                
                count += 1

    def blur_cube(self):

        self.parabole_func()
        
        for i in range(self.mbi, self.sbi):
            
            lower = 0.01
            upper = 3.00
            epsilon = 0.02

            self.edge = self.convert_coordinates_to_intensity_values(self.cube.sel(bands=i).values, self.line)
            fwhm = self.get_fwhm_val(self.edge)
            self.initial_fwhm.append(fwhm)
            
            self.current_fwhm.append(fwhm)
            self.blurred_cube = xr.DataArray(
                data=np.zeros_like(self.cube.isel(bands=slice(self.mbi, self.sbi)).values),
                coords=self.cube.isel(bands=slice(self.mbi, self.sbi)).coords,
                dims=self.cube.isel(bands=slice(self.mbi, self.sbi)).dims
            )

            while upper - lower > epsilon:
                middle = (lower + upper) / 2
                blurred_band = cv2.GaussianBlur(self.cube.sel(bands=i).values, (0,0), sigmaX=middle)
                self.blurred_cube.loc[dict(bands=i-self.mbi)] = blurred_band
                self.edge = self.convert_coordinates_to_intensity_values(self.blurred_cube.isel(bands=i-self.mbi).values, self.line)
                fwhm = self.get_fwhm_val(self.edge)
                self.current_fwhm[i-self.mbi] = fwhm

                if self.current_fwhm[i-self.mbi] > self.desired_fwhm[i-self.mbi]:
                    upper = middle
                else:
                    lower = middle

            final_sigma = (lower + upper) / 2
            self.sigma_values.append(final_sigma)
            self.blurred_cube.loc[dict(bands=i-self.mbi)] = cv2.GaussianBlur(self.cube.sel(bands=i).values, (0,0), sigmaX=self.sigma_values[i-self.mbi])
            self.final_fwhm.append(self.get_fwhm_val(self.convert_coordinates_to_intensity_values(self.blurred_cube.isel(bands=i-self.mbi).values, self.line)))

    def get_cubes(self):
        """Get the raw data from the folder.

        Returns:
            list: The list of paths to the raw data files.
        """
        # Initialize an empty list to store the paths to the .nc files
        self.paths_to_nc = []

        # Find files ending in .nc
        for i, file in enumerate(os.listdir(self.folder_name)):
            if i == 0:
                continue
            if file.endswith(".nc"):
                # Append the path to the .nc file to the list
                self.paths_to_nc.append(os.path.join(self.folder_name, file))
                print("File accessed: ", os.path.join(self.folder_name, file))

        # Return the list of paths to the .nc files
        return self.paths_to_nc

    def get_cube(self):
        """Get the raw data from the folder.

        Returns:
            np.ndarray: The raw data.
        """
        # find file ending in .nc
        for file in os.listdir(self.folder_name):
            if file == "edge_image.nc":
            #if file.endswith(".L1B_ISS"):
                self.path_to_nc = os.path.join(
                    self.folder_name, file)
                print("File accessed: ", self.path_to_nc)
                break

        # Data from wavelengths less than 400 nm and greater than 900 nm are not recommended for analysis, but we will use them anyway, we can throw data away if needed, ask sivert


    def read_cube(self):
        #f = nc.Dataset(self.path_to_nc, 'r')
        #ds = xr.open_dataset(self.path_to_nc, group='products', engine='h5netcdf', phony_dims='access')
        ds = xr.open_dataset(self.path_to_nc, group='products', engine='h5netcdf')
        print(ds.variables)
        Lt = ds['Lt']
        print(Lt)

        # Only need to do this once
        if self.wavelengths == 0:
            self.wavelengths = Lt.attrs['wavelengths']

        slope = 0.02  # The slope value mentioned in the documentation
        Lt_corrected = Lt * slope

        self.bands = len(self.wavelengths[self.mbi:self.sbi] - 1 )
        self.wavelengths = self.wavelengths[self.mbi:self.sbi]
        
        self.cube = Lt_corrected[:, :500, :1870:]
        

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
        
        self.line = [214, 701, 214, 692] # Manuel coordinates for a line crossing a sharo edge, only need to do this once
        
        # Get the pixel intensity values along the vertical line
        self.edge = self.convert_coordinates_to_intensity_values(self.cube.sel(bands=96).values, self.line)

        print("Edge:", self.edge)
        

    def visualize_cube(self):

        # Assuming 'cube' is your xarray Dataset
        R = self.cube.sel(bands=42).values
        G = self.cube.sel(bands=27).values
        B = self.cube.sel(bands=11).values

        self.line = [110, 830, 110, 823]


        # Normalize to [0, 1]
        R = (R - R.min()) / (R.max() - R.min())
        G = (G - G.min()) / (G.max() - G.min())
        B = (B - B.min()) / (B.max() - B.min())

        # Stack the R, G, B bands to create a 3D array (image)
        rgb_image = np.dstack((R, G, B))

        plt.imshow(rgb_image)
        plt.show()

    def visualize_cube_zoomed(self):
        import matplotlib.patches as patches

        image = self.cube.sel(bands=96).values
        image = image - np.min(image)  # Shift the range so that it starts from 0
        image = image / np.max(image)  # Normalize to the range [0, 1]
        image = (image * 255).astype(np.uint8)  # Scale to the range [0, 255] and convert to 8-bit integers

        # Your existing code
        self.line = [110, 830, 110, 823]
        center_x = self.line[0] + (self.line[2] - self.line[0]) // 2
        center_y = self.line[1] + (self.line[3] - self.line[1]) // 2
        zoomed_image = rgb_image[center_y-10:center_y+10, center_x-10:center_x+10]

        # Create a figure and a 1x2 grid of subplots
        fig, axs = plt.subplots(1, 2)

        
        # Draw a rectangle on the original image to indicate the zoomed area
        # Define the size of the rectangle
        rect_width = 70
        rect_height = 90

        # Calculate the bottom left corner of the rectangle
        rect_x = center_x - rect_width // 2
        rect_y = center_y - rect_height // 2

        # Create the rectangle
        rect = patches.Rectangle((rect_x, rect_y), rect_width, rect_height, linewidth=1, edgecolor='r', facecolor='none')
        axs[0].add_patch(rect)
        axs[0].set_xticks([])  # Remove x-axis ticks
        axs[0].set_yticks([])

        # Plot the original image on the first subplot
        axs[0].imshow(rgb_image, cmap='gray')


        # Your existing code for the zoomed image and stippled line
        zoomed_line_x = [self.line[0] - (center_x-10), self.line[2] - (center_x-10)]
        zoomed_line_y = [self.line[1] - (center_y-10), self.line[3] - (center_y-10)]
        zoomed_line_x = [max(min(x, 50), 0) for x in zoomed_line_x]
        zoomed_line_y = [max(min(y, 50), 0) for y in zoomed_line_y]
        axs[1].set_xticks([])  # Remove x-axis ticks
        axs[1].set_yticks([])
        axs[1].imshow(zoomed_image, cmap='gray')
        axs[1].plot(zoomed_line_x, zoomed_line_y, 'r--')

        # Display the figure
        plt.show()
        
    def plot_edge_fwhm(self):
        """
            Plotting the found FWHM values from the ESF. 
        """
        plt.plot(self.wavelengths, self.final_fwhm)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('FWHM spatial pixels')

        plt.show()
    def plot_edge_fwhm(self):
        """
            Plotting the found FWHM values from the ESF. 
        """
        plt.plot(self.wavelengths, self.initial_fwhm)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('FWHM spatial pixels')

        plt.show()

    def parabole_func(self):
        bands = self.bands
        a_1 = 4 * (self.blurriest_fwhm - self.sharpest_fwhm) / ((bands - 1) ** 2)
        for band in range(bands):  # Iterate over the range of bands
            self.desired_fwhm.append(a_1 * (band - (bands - 1) // 2) ** 2 + self.sharpest_fwhm)

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




        



        


