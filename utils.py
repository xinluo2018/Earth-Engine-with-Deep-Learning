#!/usr/bin/env python
# coding: utf-8

# In[12]:


# mount on google drive
from google.colab import drive
drive.mount('/content/drive/')
# go to your work patch
import os
os.chdir("/content/drive/My Drive/Earth-Engine-with-Deep-Learning")
#!ls
# !nvidia-smi


# In[13]:


from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt


# In[14]:


### tiff image reading
def readTiff(path_in):
    '''
    return: numpy array, dimentions order: (row, col, band)
    '''
    RS_Data=gdal.Open(path_in)
    im_col = RS_Data.RasterXSize  # 
    im_row = RS_Data.RasterYSize  # 
    im_bands =RS_Data.RasterCount  # 
    im_geotrans = RS_Data.GetGeoTransform()  # 
    im_proj = RS_Data.GetProjection()  # 
    RS_Data = RS_Data.ReadAsArray(0, 0, im_col, im_row)  # 
    if im_bands > 1:
        RS_Data = np.transpose(RS_Data, (1, 2, 0)).astype(np.float)  # 
        return RS_Data, im_geotrans, im_proj, im_row, im_col, im_bands
    else:
        return RS_Data,im_geotrans,im_proj,im_row,im_col,im_bands


# In[15]:


def imgShow(img, num_bands, clip_percent):
    '''
    Arguments: 
        img: (row, col, band) or (row, col)
        num_bands: a list/tuple, [red_band,green_band,blue_band]
        clip_percent: for linear strech, value within the range of 0-100. 
    '''
    if len(img.shape) == 2:
        img = img[:,:,np.newaxis]
    img_color = img[:,:,[num_bands[0], num_bands[1], num_bands[2]]]
    where_are_NaNs = np.isnan(img_color)
    img_color[where_are_NaNs] = 0
    img_color_nor = (img_color-np.min(img_color))/(np.max(img_color)-np.min(img_color))
    img_color_nor_hist = np.percentile(img_color_nor, [clip_percent, 100-clip_percent])
    img_color_nor_clip = (img_color_nor-img_color_nor_hist[0])/(img_color_nor_hist[1]-img_color_nor_hist[0])
    
    plt.imshow(img_color_nor_clip)


# In[16]:


## convert model.ipynb to model.py
try:
    get_ipython().system(u'jupyter nbconvert --to python utils.ipynb')
except:
    pass

