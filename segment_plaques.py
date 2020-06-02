#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Script to segment plaques by using watershed, 
# it uses Pythton 3.x

from skimage import morphology
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
#from skimage import img_as_float
from skimage import exposure,io  
from skimage.filters import  threshold_otsu  #, threshold_niblack,rank
import numpy as np
import  tifffile 
from joblib import Parallel, delayed 
import sys
from skimage.filters import frangi
#from skimage.measure import regionprops , label
from  scipy.ndimage.measurements import label 


#Settings 
# Load an example image
input_namefile = sys.argv[1]
output_namefile = 'seg_'+ input_namefile
smallest_area = 20
tubular_removal = True
#block_size = 51 #Size block of the local thresholding 

#Functions 
def thresh(arr):

    #By default the local thresholding is according to the mode of the Gaussian
    #local threshold
    thresh = threshold_otsu(arr)
#    thresh = threshold_niblack(arr, window_size=block_size , k=0.01) # threshold_local(img[i,:,:],block_size, offset=10)
    binary_adaptive = arr > thresh
    res = np.asanyarray(binary_adaptive)
    tifffile.imsave('neg_' + output_namefile, res, bigtiff=True)
    return res

def ws(binary_adaptive):
    # denoised = denoise_tv_chambolle(cleaned, weight=0.2, multichannel=False )
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance
    # to the background
    distance = ndimage.distance_transform_edt(binary_adaptive)
    #Footprint is one of the parameters which influence over- or under-segmentation, change it if necessary
    local_maxi = peak_local_max( distance, indices=False, footprint=np.ones((15, 15)), labels= binary_adaptive ) #
    markers = ndimage.label(local_maxi)[0]
    labels =  watershed(-distance, markers, mask= binary_adaptive, watershed_line=True )
    return labels 
 
print('Loading')
img = io.imread(input_namefile, plugin='tifffile') 
print("loaded")

# This sends multiple jobs for segmentation using parallelization
res = Parallel(n_jobs=16, backend="threading")(delayed(thresh)(i) for i in img)
res = np.asanyarray(res,dtype=np.int16)
print("saving segmentation")
tifffile.imsave('neg_' +output_namefile, res, bigtiff=True)

#Remove tubular structures
if tubular_removal == True: 
    d, r, c, = np.shape(res)
    frangi_res  = np.zeros((d,r,c))
    for i in range(d):
        frangi_res[i,:,:] =  frangi(res[i,:,:],  black_ridges=False ,beta=0.1)
    print("saving Frangi detects")     
    frangi_res = frangi_res > 2E-14
    #Threshold not set to 0 as sometimes due to numerical issues the Frangi detector is showing some 0 values as really small
    frangi_res = np.asanyarray(frangi_res,np.int16)
    #tifffile.imsave('Frangi_'+output_namefile, frangi_res, bigtiff=True)
    res = res - frangi_res
    res = res > 0 
    #Remove noise post-tubular removal
    selem = morphology.ball(1)
    res = morphology.binary_erosion(res, selem)
    res = morphology.remove_small_objects(res, smallest_area)
    # IS DILATION NECESSARY?!?!?!? If Annamaria happy with results no.
    res = np.asanyarray(res,dtype=np.int16)
    #tifffile.imsave('sub'+output_namefile, res, bigtiff=True)
 
# This sends multiple jobs for watersheding using parallelization
res = Parallel(n_jobs=16, backend="threading")(delayed(ws)(i) for i in res)

# Extract the region props of the
#label_img =  label(res)
#props = regionprops(label_img)
#res = np.array([prop.label for prop in props])
res = np.asanyarray(res,dtype=np.int16)
#res = res > 0 

#res = np.asanyarray(res)
print("saving segmentation WS")
#tifffile.imsave(output_namefile, res, bigtiff=True)

#Kernel connectivity
str_3D = np.ones((3,3,3))
# If left  as ones(3,3,3) it means in 3D: 
# \  |  /
# -  x  -
# /  |  \ 

res, num_features  = label(res,structure=str_3D)

#res = measure.label(res,connectivity=3)
res = np.asanyarray(res,dtype=np.int16)
tifffile.imsave(output_namefile, res, bigtiff=True)

# Total number of plaques
tot = num_features# np.amax(res.flat)
print('Number of segmented plaques')
print(tot)

#Compute density
our_convex_hull  = np.zeros((d,r,c))
for i in range(d):
     our_convex_hull[i,:,:] =  morphology.convex_hull_image(res[i,:,:])
tifffile.imsave('CH_'+output_namefile, our_convex_hull, bigtiff=True)

vol_size = np.sum(our_convex_hull.flat)
dens = tot/vol_size
print('The density of the plqeue is ')
print(dens)
