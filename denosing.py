# -*- coding: utf-8 -*-
"""
Created on Tue May  29 17:31:54 2020

@author: josh-
"""


from skimage import io, img_as_float
from scipy import ndimage as nd
from matplotlib import pyplot as plt
import numpy as np

##1st filter gaussian filter

img = img_as_float(io.imread("images/1884865667.jpg"))
guassian_img = nd.gaussian_filter(img, sigma=3)

plt.imsave("images/gaussian_1884865667.jpg", guassian_img)

##image at this satge is clean(low noise) but blurry

##2nd filter median filter

median_img = nd.median_filter(img, size=3)
plt.imsave("images/median_1884865667.jpg", median_img)

##3rd filter non-local mean denosing

from skimage.restoration import denoise_nl_means, estimate_sigma

sigma_est = np.mean(estimate_sigma(img, multichannel=True))
nlm = denoise_nl_means(img, h=1.15*sigma_est, fast_mode=True, patch_size=5, patch_distance=3, multichannel=True)

plt.imsave("images/nlm_1884865667.jpg", nlm)