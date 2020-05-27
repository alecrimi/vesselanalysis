#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Script to segment plaques by using watershed, 
# it uses Pythton 3.x

from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
from skimage import img_as_float
from skimage import exposure,io 
from skimage import external 
from skimage.color import rgb2gray
import numpy as np
from skimage.filters import threshold_local

from joblib import Parallel, delayed

import sys

# Load an example image
input_namefile = sys.argv[1]
output_namefile = 'seg_'+ input_namefile  

#Settings 
block_size = 35  #Size block of the local thresholding 

#Functions 
def f(arr):

    #By default the local thresholding is according to the mode of the Gaussian
    binary_adaptive = threshold_local(arr, block_size, offset=10)

    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance
    # to the background
    distance = ndimage.distance_transform_edt(binary_adaptive)
    local_maxi = peak_local_max( distance, indices=False, footprint=np.ones((3, 3)), labels=binary_adaptive)
    markers = ndimage.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=binary_adaptive)

    #img_adapteq = exposure.equalize_adapthist(arr, clip_limit=0.01) #  kernel_size=k,
    #img_adapteq = img_adapteq.astype(np.float16)
    #img_gray = rgb2gray(img_adapteq)     
    return labels

print('Loading')
img = io.imread(input_namefile, plugin='tifffile') 
print("loaded")

# This sends multiple jobs using parallelization
res = Parallel(n_jobs=8, backend="threading")(delayed(f)(i) for i in img)
#res = f( img)
res = np.asanyarray(res)

print("saving segmentation")
external.tifffile.imsave(output_namefile, res,bigtiff=True) #,plugin="tifffile"
#external.tifffile.imsave('small'+output_namefile, res,bigtiff=True, compress='LZMA') #,plugin="tifffile"
