import requests

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


class blurredImage:
    def __init__(self) -> None:
        self.image_height = None
        self.image_width = None
        self.bands = None
        self.image = None
    
    def get_hico_images(self):
        self.raw_image = hicodata
        image = hicodata.getdata
        


