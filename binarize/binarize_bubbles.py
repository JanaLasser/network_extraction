# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:26:39 2015

@author: jana
"""
import os
from os import getcwd
from os.path import join
import sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../net'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

from PIL import Image
import numpy as np
from scipy.misc import imsave
from skimage.filters import sobel
from skimage.morphology import remove_small_objects
from net_helpers import RGBtoGray

cwd = getcwd()
src = join(cwd.split('binarize')[0],join('data',join('originals','bubbles')))
dest = join(cwd.split('binarize')[0],join('data','binaries'))
    
bubbles = np.asarray(Image.open(join(src,'bubbles1.png'))) 
bubbles = bubbles.astype(np.uint8)
bubbles = bubbles[0:,0:,1]
background = np.asarray(Image.open(join(src,'background.png')))
background = background.astype(np.uint8)
background = RGBtoGray(background)
wire = np.asarray(Image.open(join(src,'wire.png')))
wire = wire.astype(np.uint8)
wire = RGBtoGray(wire)

bubbles = np.where(background < 255, 255, bubbles)
bubbles = np.where(wire < 200, 0, bubbles)
edges = sobel(bubbles)

bubbles = np.where(bubbles > 125,0,1)
edges = np.where(edges > 0.1,1,0)
edges = np.where(wire < 200, 1, edges)
edges = edges.astype(np.bool)
edges = remove_small_objects(edges,10000)

imsave(join(dest,'bubbles1_binary.png'),edges)



