# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:44:59 2015

@author: jana
"""

import vectorize_helpers as vh
import cv2
import numpy as np


def getContours(image):
    image = image.astype(np.uint8)
    contours,hirachy = cv2.findContours(image,cv2.RETR_CCOMP,\
                                        cv2.CHAIN_APPROX_TC89_L1)
    contours = np.asarray(contours)   
    
    return (contours,hirachy)

image = vh.getImage("data/contour_test.png")
h,w = image.shape
raw_contours, hirachy = getContours(image)
flattened_contours = vh.flattenContours(raw_contours)

#find longest
length = 0
index = 0
for i,c in enumerate(flattened_contours):
    if len(c) > length:
        index = i
        
filtered_contours = []
for i,c in enumerate(flattened_contours):
    if (hirachy[0][i][2] == -1 and i != index):
        pass
    else:
        filtered_contours.append(c)

vh.drawContours(h,flattened_contours,"test_contours","data")