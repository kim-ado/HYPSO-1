import urllib
import os
from html.parser import HTMLParser
import requests
import bz2

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
                        if value.endswith('/'):  # Check if the link is a subdirectory
                            self.dataList.append(value)  # Store the subdirectory
                    
    def handle_endtag(self, tag):
            if tag == 'table':
                self.Counter +=1

    def handle_data(self, data):
        if self.Counter == 1:
            if self.lasttag == 'a' and self.inLink and data.strip():
                self.dataList.append(data)

    def reset_state(self):
        self.inLink = False
        self.dataList = []
        self.Counter = 0
        

# Define function for batch downloading
def BatchJob(Files, cookie_jar, url, folder_name):
    for dat in Files:
        if dat.endswith('.bz2'):  # Check if the file is a .bz2 file
            print("downloading: ", dat)
            JobRequest = urllib.request.Request(url+dat)
            JobRequest.add_header('cookie', str(cookie_jar)) # Pass the saved cookie into additional HTTP request
            JobResponse = urllib.request.urlopen(JobRequest)

            JobRedirect_url = JobResponse.geturl() + '&app_type=401'
            # Request the resource at the modified redirect url
            Request = urllib.request.Request(JobRedirect_url)
            Response = urllib.request.urlopen(Request)
            bz2_file = folder_name + '/' + dat
            with open(bz2_file, 'wb') as f:  # Add the path to the hico_data folder
                f.write(Response.read())
            Response.close()

            # Unzip the .bz2 file
            newfilepath = bz2_file[:-4]
            with open(newfilepath, 'wb') as new_file, bz2.BZ2File(bz2_file, 'rb') as file:
                for data in iter(lambda : file.read(100 * 1024), b''):
                    new_file.write(data)

            # Delete the .bz2 file
            os.remove(bz2_file)

    print("Files downloaded and extracted to: ", os.path.dirname(os.path.realpath(__file__)) + '/' + folder_name)






