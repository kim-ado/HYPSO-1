# Hyperspectral Image Sharpening for HYPSO-1 Data
This repo is a collection of some code that can be used to evaluate and test methods for image sharpening on hyperspectral data collected by minitiaurized push-broom sensors.

#### Get Started
Make sure you have a correct python installment. Assuming you have [conda](https://docs.conda.io/en/latest/) installed simply run
```
conda env create --file environment.yml
```
in the terminal to create the envirnoment. Followed by
```
conda activate h1-sharp
```
to enter into this new envirnoment

You should also make sure that you have been able to install the correct submodules. This can be done by typing
```
git submodule update --init --recursive
git submodule foreach --recursive git fetch
git submodule foreach git merge origin master
```

in your terminal. if you then have a folder named `cal-char-corr` you have been sucessful.

#### Get the data
In the renders given in notebooks here the datasets 
```
20221010_CaptureDL_tampa_2022_10_08T15_39_46
20221107_CaptureDL_sudan_tl_2022_11_04T08_31_09
20221027_CaptureDL_bangladesh_2022_10_26T04_02_54
20221201_CaptureDL_glacierbay_2022_11_30T20_18_05
20221028_CaptureDL_bangladesh_2022_10_27T03_50_36
20221205_CaptureDL_florida_2022_12_02T15_16_12
20221107_CaptureDL_san_francisco_tr_2022_11_03T18_10_46
```
are used for the analysis. simply go to [this](https://studntnu.sharepoint.com/:f:/s/o365_HYPSO-project/EgMocI3tNOhFoOmmnHpaydsB7qR5AzeaYjTua7sn5VRn-w?e=ycyZoo) to find the data sets or new data sets that you want to use. The given link requires that you are able to access NTNU servers.

#### Repo Structure
This repo does contain some files fora manuscript in addtion to a notebook file `notes.ipynb` and some utility functions in `sharp.py`. If the conda envirnoment has been installed and activated corretly there should be no problems with running the notebook.