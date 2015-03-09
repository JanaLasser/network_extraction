# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:32:18 2013

@author: Ista
"""
from os.path import join
import cv2
from skimage.morphology import disk,closing, remove_small_objects,opening
from skimage.filter import rank
import numpy as np
import scipy.misc
import vectorize_helpers as vh
import argparse
import ntpath

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
image = vh.getImage(image_source)
image = vh.RGBtoGray(image)
image = image.astype(np.uint8)
					
#blur image to get rid of most of the noise
blur_kernel = (3,3)
image = cv2.blur(image, blur_kernel)

#perform local histogram equalization to emphasize edges
selem = disk(11)
image_eq = rank.equalize(image, selem=selem)

#add half of the processed and half of the original image to further separate
#regions where there is structure from the background
image = 0.5*image_eq + 0.5*image

#find a favorable threshold using otsu thresholding and modify it by t_mod
threshold = vh.otsu_threshold(image)-t_mod

#threshold and save image
image = np.where(image > threshold,1.0,0.0)
image = opening(image,disk(3))
image = closing(image,disk(7))
image = remove_small_objects(image.astype(bool),min_size=50,connectivity=1)


scipy.misc.imsave(join(dest,image_name + "_binary.png"),image)
print "image processed with t_mod = %d" %t_mod




