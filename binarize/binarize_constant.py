'''
    Copyright (C) 2015  Jana Lasser Max Planck Institute for Dynamics and
    Self Organization Goettingen

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    If you find any bugs or have any suggestions, please contact me at
    jana.lasser@ds.mpg.de
'''

#path handling
from os.path import join
import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../net'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

import cv2
from cv2 import imread, blur, threshold
from skimage.morphology import disk,closing, remove_small_objects, \
    remove_small_holes, binary_opening, binary_closing

import numpy as np
from scipy.misc import imsave
from net_helpers import RGBtoGray
import argparse
from ntpath import basename

'''
Argument handling:
------------------

Required: source
        The program will load the image specified in "source".
        The image needs to be an 8-bit image.
        
Optional: destination -dest
        The script will save all results in "destination" if it is specified.
        Otherwise results are saved at the destination of the source image.
        
Optional: threshold_modifier -t
        The correct threshold for the thresholding process will be determined
        with Otsu's method. Nevertheless sometimes this process does not yield
        the best results and the threshold needs some manual tweaking. When
        processing a dataset, just look at the first few images, see if the
        binaries look good and if not, tweak the threshold a bit via the -t
        modifier.
        
Optional: minimum_feature_size -s
        If minimum_feature_size in pixels is specified, features with a size of
        up to minimum_feature_size will be removed from the binary image
        previous to processing. This serves to a) reduce the number of contours
        which will be processed and not needed in the end and b) helps the
        triangulation algorithm to not get confused when there are islands
        within holes.
        Defaults to 3000 pixels.

Optional: image_improvement -i
        If image_improvement is enabled, the binary image will be smoothed
        using binary opening and closing operations. The kernel size of said
        operations can be controlled by passing an integer along with the
        image_improvement option. The right kernel size is highly dependant on
        the resolution of the image. As a rule of thumb choose a kernel size
        below the width of the smallest feature in the image (in pixels).
        Defaults to 3 pixels.
'''


#argument handling
parser = argparse.ArgumentParser(description='Binarize: a script that ' +\
    'performs some basic image enhancement operations to sharpen contours'\
    + ' and increase contrast as well as thresholding to create a binary image.' \
    + '\nCopyright (C) 2015 Jana Lasser')
    
parser.add_argument('source',type=str,help='Complete path to the image')

parser.add_argument('-dest', type=str, help='Complete path to the folder '\
                + 'results will be saved to if different than source folder')

parser.add_argument('-g','--gaussian_blur', type=int, \
                help='Kernel size for the Gaussian blur.', default=3)
                
parser.add_argument('-t','--threshold_offset', type=int, \
                help='Modifier for the selected threshold', default=0)
                
parser.add_argument('-m','--minimum_feature_size', type=int, \
                help='Minimum size (pixels) up to which features will be ' + \
                'discarded.', default = 3000)

parser.add_argument('-s','--smoothing',type=int,\
                    help='Sets kernel size of the binary opening and closing'+\
                    ' operators.', default=3)
                
parser.add_argument('-i','--invert',action='store_true',\
                    help='Inverts the intensity of the image.')

args = parser.parse_args()
image_source = args.source
dest = args.dest
threshold_offset = args.threshold_offset
minimum_feature_size = args.minimum_feature_size                               #features smaller than minimum_feature_size will be discarded.
smoothing = args.smoothing                                                     #enables smoothing via binary opening and closing on and off
gaussian_blur = args.gaussian_blur
invert = args.invert

image_name,ending = basename(image_source).split('.')
if dest == None:
    dest = image_source.split(image_name + '.' + ending )[0]

#load the image
image = imread(image_source)
image = RGBtoGray(image)
image = image.astype(np.uint8)

#invert image if necessary
if invert:
    image = 255 - image
					
#blur image to get rid of most of the noise
if gaussian_blur > 0:
    blur_kernel = (gaussian_blur,gaussian_blur)
    image = blur(image, blur_kernel)

#calculate Otsu's threshold
threshold, ret = threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
threshold -= threshold_offset

#threshold image
image = np.where(image > threshold, 1,0)

#remove disconnected objects
image = remove_small_objects(image.astype(bool),\
    min_size=minimum_feature_size,connectivity=1)

#smoothe with binary opening and closing
if smoothing:                                                                  #standard binary image noise-removal with opening followed by closing
    image = binary_opening(image,disk(smoothing))                              #maybe remove this processing step if depicted structures are really tiny
    image = binary_closing(image,disk(smoothing))  

#remove disconnected objects and fill in holes
image = remove_small_objects(image.astype(bool),\
    min_size=minimum_feature_size,connectivity=1)
image = remove_small_holes(image.astype(bool),\
    min_size=minimum_feature_size/100,connectivity=1)

#save image
imsave(join(dest,image_name + "_binary.png"),image)





