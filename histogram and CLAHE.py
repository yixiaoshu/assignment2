# -*- coding: utf-8 -*-
"""
Created on Tue May  27 17:31:54 2020

@author: josh-
"""

import cv2
from skimage import io
from matplotlib import pyplot as plt

img = cv2.imread("images/nlm_1884865667.JPG", 1)

lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

l, a, b = cv2.split(lab_img)
#1
plt.hist(l.flat, bins=100, range=(0, 255))
equ = cv2.equalizeHist(l)

#2 after histogram equalization
#plt.hist(equ.flat, bins=100, range=(0, 255))
#plt.imshow(equ, cmap='gray')

updated_lab_img1 = cv2.merge((equ,a,b))
hist_eq_img = cv2.cvtColor(updated_lab_img1, cv2.COLOR_BGR2LAB)

###########CLAHE#########################
#Apply CLAHE to L channel
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
clahe_img = clahe.apply(l)
plt.hist(clahe_img.flat, bins=100, range=(0,255))

#Combine the CLAHE enhanced L-channel back with A and B channels
updated_lab_img2 = cv2.merge((clahe_img,a,b))

#Convert LAB image back to color (RGB)
CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)


cv2.imshow("Original image", img)
cv2.imshow("Equalized image", hist_eq_img)
cv2.imshow('CLAHE Image', CLAHE_img)

cv2.waitKey(0)

cv2.destroyAllWindows() 