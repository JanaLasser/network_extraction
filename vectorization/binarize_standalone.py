# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:32:18 2013

@author: Ista
"""
from os.path import join
import cv2
from skimage.morphology import disk,closing, remove_small_objects,opening
from skimage.filter import rank
from PIL import Image
import numpy as np
import scipy.misc
import argparse
import ntpath

#necessary functions
def getImage(image_path):
    image = Image.open(image_path)
    image = image.convert('L')
    image = np.asarray(image,dtype=np.uint8)
    return image

def RGBtoGray(image):
    if(isinstance(image,str)):
        image = getImage(image)
    if (image.ndim == 2): return image      
    else:
        image = image.astype(np.float32).sum(axis=2);image/=3.0
        return image

def otsu_threshold(image):

    histData = np.histogram(image, np.arange(256))
    histData = np.asarray(histData[0])
    l = 256 - len(histData)
    while(l != 0):
        histData = np.append(histData,0)
        l -=1
    total = image.size
    sum = 0.0
    t = 0

    while(t < 256):
        sum += t*float(histData[t])
        t +=1

    sumB = 0.0
    wB = 0
    wF = 0
    varMax = 0.0
    threshold = 0
    t = -1

    while(t < 256):
        t += 1
        wB += histData[t]
        if(wB == 0):
            continue
        wF = total - wB;
        if(wF == 0):
            break
            
        sumB += float(t)*float(histData[t])
        mB = float(sumB) / float(wB)
        mF = float(sum - sumB) / float(wF)
        varBetween = float(wB) * float(wF) * (float(mB) - float(mF)) * (float(mB) - float(mF))
        if (varBetween > varMax):
            varMax = varBetween
            threshold = t
            
    return threshold

#argument handling
parser = argparse.ArgumentParser(description='Binarize: a script that ' +\
    'performs some basic image enhancement operations to sharpen contours'\
    + ' and increase contrast as well as thresholding to create a binary image.' \
    + '\nCopyright (C) 2015 Jana Lasser')
parser.add_argument('source',type=str,help='Complete path to the image')
parser.add_argument('-dest', type=str, help='Complete path to the folder '\
                + 'results will be saved to if different than source folder')
parser.add_argument('-t','--threshold_modifier', type=int, \
                help='Modifier for the selected threshold', default=0)
                
args = parser.parse_args()
image_source = args.source
dest = args.dest
t_mod = args.threshold_modifier

image_name,ending = ntpath.basename(image_source).split('.')
if dest == None:
    dest = image_source.split(image_name + '.' + ending )[0]

#load the image
image = getImage(image_source)
image = RGBtoGray(image)
image = image.astype(np.uint8)
					
#blur image to get rid of most of the noise
blur_kernel = (5,5)
image = cv2.blur(image, blur_kernel)

#perform local histogram equalization to emphasize edges
selem = disk(11)
image_eq = rank.equalize(image, selem=selem)

#add half of the processed and half of the original image to further separate
#regions where there is structure from the background
image = 0.5*image_eq + 0.5*image

#find a favorable threshold using otsu thresholding and modify it by t_mod
threshold = otsu_threshold(image)-t_mod

#threshold and save image
image = np.where(image > threshold,1.0,0.0)
image = opening(image,disk(3))
image = closing(image,disk(7))
image = remove_small_objects(image.astype(bool),min_size=50,connectivity=1)

scipy.misc.imsave(join(dest,image_name + "_binary.png"),image)
print "image processed with t_mod = %d" %t_mod




