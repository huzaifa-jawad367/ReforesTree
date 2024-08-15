import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.plot import reshape_as_image
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = None
import sys
package = os.path.dirname(os.getcwd())
sys.path.append(package)
sys.path.append(package + 'utils')

#import seaborn as sns
#colors = sns.color_palette('tab10')
#mypalette={'NN':colors[0], 'GMN':colors[4], 'OT':colors[1], 'OT on GPS position':colors[1], 'GW':colors[2], 'OT on GPS position + Tree species':colors[3]}
#import matplotlib.pylab as plt
#import tensorflow

import warnings
warnings.filterwarnings('ignore')

from utils.extract_features import *
from utils.deepforest_detection import *
#from utils.visualisation import *
#from utils.plot_folium import *
#from utils.plot_density import *
#from utils.mapping import *

directory = "data/wwf_ecuador/RGB Orthomosaics"
save_dir = "data/tiles"

# Extracting the main information for each site 
ortho_data = create_ortho_data(directory, os.path.join(save_dir, 'ortho_data.csv'))

# print(ortho_data.head())
print(ortho_data.info())
# print(ortho_data.describe())

# Split images into tiles (might takes some minutes)
for file in os.listdir(directory):
    if file.endswith('.tif'):
        # Open image file for reading (binary mode)
        path_to_raster = os.path.join(directory, file)
        name = file.replace('.tif', '')

        tiles_dir = os.path.join(save_dir, name)
        if not os.path.exists(tiles_dir):
            os.makedirs(tiles_dir)        
            split_raster(path_to_raster, base_dir=tiles_dir, patch_size=4000, patch_overlap=0.05)
            
