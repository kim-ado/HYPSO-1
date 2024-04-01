import requests
import cv2
import numpy as np
import os
import xarray as xr
import numpy as np
import scipy.interpolate as si
import scipy.optimize as so
import urllib
import os
from http.cookiejar import CookieJar
from html.parser import HTMLParser

'''
 This script, NSIDC_parse_HTML_BatchDL.py, defines an HTML parser to scrape data files from 
 an earthdata HTTPS URL and bulk downloads all files to your working directory.

 This code was adapted from https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+Python

 Last edited Sep 12, 2022
 Tested on Python 3
 
 ===============================================
 Technical Contact
 ===============================================

 NSIDC User Services
 National Snow and Ice Data Center
 CIRES, 449 UCB
 University of Colorado
 Boulder, CO 80309-0449  USA
 phone: +1 303.492.6199
 fax: +1 303.492.2468
 form: Contact NSIDC User Services
 e-mail: nsidc@nsidc.org

'''

#===============================================================================
# Call the function to download all files in url
#===============================================================================

#BatchJob(Files, cookie_jar) # Comment out to prevent downloading to your working directory

class hicoData:
    def __init__(self):
        self.folder_name = "C:/Users/Kim/Documents/master/HYPSO-1/hico_data"
        files_folder = None
        data = None
        cube = []
        self.sbi = 750 # The center wavelength of the band of interest

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

    # Define a custom HTML parser to scrape the contents of the HTML data table
    class MyHTMLParser(HTMLParser):
        def __init__(self):
            HTMLParser.__init__(self)
            self.inLink = False
            self.dataList = []
            self.directory = '/'
            self.indexcol = ';'
            self.Counter = 0
            
        def handle_starttag(self, tag, attrs):
            self.inLink = False
            if tag == 'table':
                self.Counter += 1
            if tag == 'a':
                for name, value in attrs:
                    if name == 'href':
                        if self.directory in value or self.indexcol in value:
                            break
                        else:
                            self.inLink = True
                            self.lasttag = tag
                        
        def handle_endtag(self, tag):
                if tag == 'table':
                    self.Counter +=1

        def handle_data(self, data):
            if self.Counter == 1:
                if self.lasttag == 'a' and self.inLink and data.strip():
                    self.dataList.append(data)
            
            
    def https_authentication(self):
        #===============================================================================
        # The following code block is used for HTTPS authentication
        #===============================================================================

        # The user credentials that will be used to authenticate access to the data
        username = "YOUR_USERNAME"
        password = "YOUR_PASSWORD"

        # The FULL url of the directory which contains the files you would like to bulk download
        url = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0192_seaice_trends_climo_v3/total-ice-area-extent/nasateam/" # Example URL

        # Create a password manager to deal with the 401 reponse that is returned from
        # Earthdata Login
        
        password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_manager.add_password(None, "https://urs.earthdata.nasa.gov", username, password)
        
        # Create a cookie jar for storing cookies. This is used to store and return
        # the session cookie given to use by the data server (otherwise it will just
        # keep sending us back to Earthdata Login to authenticate).  Ideally, we
        # should use a file based cookie jar to preserve cookies between runs. This
        # will make it much more efficient.
        
        cookie_jar = CookieJar()

        # Install all the handlers.
        opener = urllib.request.build_opener(
            urllib.request.HTTPBasicAuthHandler(password_manager),
            #urllib.request.HTTPHandler(debuglevel=1),    # Uncomment these two lines to see
            #urllib.request.HTTPSHandler(debuglevel=1),   # details of the requests/responses
            urllib.request.HTTPCookieProcessor(cookie_jar))
        urllib.request.install_opener(opener)
        
        # Create and submit the requests. There are a wide range of exceptions that
        # can be thrown here, including HTTPError and URLError. These should be
        # caught and handled.

        #===============================================================================
        # Open a requeset to grab filenames within a directory. Print optional
        #===============================================================================

        DirRequest = urllib.request.Request(url)
        DirResponse = urllib.request.urlopen(DirRequest)

        # Get the redirect url and append 'app_type=401'
        # to do basic http auth
        DirRedirect_url = DirResponse.geturl()
        DirRedirect_url += '&app_type=401'

        # Request the resource at the modified redirect url
        DirRequest = urllib.request.Request(DirRedirect_url)
        DirResponse = urllib.request.urlopen(DirRequest)

        DirBody = DirResponse.read()

        # Uses the HTML parser defined above to pring the content of the directory containing data

        parser = self.MyHTMLParser() 
        parser.feed(str(DirBody))
        Files = parser.dataList

        # Display the contents of the python list declared in the HTMLParser class
        # print Files #Uncomment to print a list of the files

    
    # Define function for batch downloading
    def BatchJob(Files, cookie_jar):
        for dat in Files:
            print("downloading: ", dat)
            JobRequest = urllib.request.Request(url+dat)
            JobRequest.add_header('cookie', str(cookie_jar)) # Pass the saved cookie into additional HTTP request
            JobResponse = urllib.request.urlopen(JobRequest)

            JobRedirect_url = JobResponse.geturl() + '&app_type=401'
            # Request the resource at the modified redirect url
            Request = urllib.request.Request(JobRedirect_url)
            Response = urllib.request.urlopen(Request)
            f = open(dat, 'wb')
            f.write(Response.read())
            f.close()
            Response.close()
        print("Files downloaded to: ", os.path.dirname(os.path.realpath(__file__)))    
    


class blurCube():
    def __init__(self):
        self.sbi = None
        self.hico_image_edge = None

        self.current_fwhm = []
        self.desired_fwhm = []

        self.blurriest_fwhm = 3.5
        self.sharpest_fwhm = 1.5

        self.guessed_sigma = []
        self.sigma_values = [] # Temp storage for sigma values to use in the GaussianBlur function

        self.cube = None
        self.blurred_cube = None

    def blur_cube(self, cube):

        self.desired_fwhm = self.generate_desired_fwhm()

        try:
            if not self.edge:
                self.edge = self.detect_sharpest_edge(cube[self.sbi]) # Finding the sharpest edge of the image at the center wavelength
        except Exception as e:
            print("Error occurred while detecting the sharpest edge:", str(e))

        for i in range(cube.shape[2]):
            
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
            self.blurred_cube[i] = cv2.GaussianBlur(self.cube[i], (5,5), sigma=final_sigma)
            self.sigma_values.append(final_sigma)
    
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
        print(data)

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




        



        


