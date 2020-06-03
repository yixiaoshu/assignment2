# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 12:12:52 2020

@author: josh-
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_ubyte, img_as_float, io
#read the image and convert to grayscale
RGBimg = io.imread("images/clahe1884865667.jpg")
img = cv2.cvtColor(RGBimg, cv2.COLOR_BGR2GRAY)
#plt.imshow(img)

#since the image we read is not in float type, need to convert
float_img = img_as_float(img)
#use nlm to denoise the image
sigma_est = np.mean(estimate_sigma(float_img, multichannel=True))

denoise_img = denoise_nl_means(float_img, h=1.15 * sigma_est, fast_mode=False, patch_size=5, patch_distance=3, multichannel=True)

denoise_ubyte = img_as_ubyte(denoise_img)
#the image now is 8-bits grayscale
#plt.imshow(denoise_ubyte, cmap='gray')

#plot the histogram
#".flat" flatten the 2D array in one dimension
#plt.hist(denoise_ubyte.flat, bins=100, range=(0, 255))

#at ths point, we can divide the segments by their concentrated range
#this generate 4 different images all in binary
segm1=(denoise_ubyte <= 50)
segm2=(denoise_ubyte > 50) & (denoise_ubyte <= 120)
segm3=(denoise_ubyte > 120) 
#create an image the same size as the input denoise_ubyte image
all_segments = np.zeros((denoise_ubyte.shape[0], denoise_ubyte.shape[1], 3)) #nothing but denoise img size but blank

all_segments[segm1] = (1,0,0)
all_segments[segm2] = (0,1,0)
all_segments[segm3] = (0,0,1)

#plt.imshow(all_segments)

#clean the images
from scipy import ndimage as nd

segm1_opened = nd.binary_opening(segm1, np.ones((3, 3)))
segm1_closed = nd.binary_closing(segm1_opened, np.ones((3, 3)))

segm2_opened = nd.binary_opening(segm2, np.ones((3,3)))
segm2_closed = nd.binary_closing(segm2_opened, np.ones((3,3)))

segm3_opened = nd.binary_opening(segm3, np.ones((3,3)))
segm3_closed = nd.binary_closing(segm3_opened, np.ones((3,3)))

all_segments_cleaned = np.zeros((denoise_ubyte.shape[0], denoise_ubyte.shape[1], 3)) #nothing but 714, 901, 3

all_segments_cleaned[segm1_closed] = (1,0,0)
all_segments_cleaned[segm2_closed] = (0,1,0)
all_segments_cleaned[segm3_closed] = (0,0,1)

plt.imshow(all_segments_cleaned)  #All the noise should be cleaned now
plt.imsave("images/segmented1884865667.jpg", all_segments_cleaned)