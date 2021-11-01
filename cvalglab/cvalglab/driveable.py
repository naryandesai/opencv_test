import cv2
import imutils
import numpy as np
from skimage.measure import ransac, LineModelND
import math
import imagefilters
import random
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from skimage import data
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.util import img_as_ubyte
import glob
import imutils
import imagefilters
import datetime
import os
import math


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def convert_bgr2c1c2c3(img):
	im = img.astype(np.float32)+0.001 #to avoid division by 0
	return np.arctan(im / np.dstack((
	    cv2.max(im[..., 1], im[..., 2]),
	    cv2.max(im[..., 0], im[..., 2]),
	    cv2.max(im[..., 0], im[..., 1]),
	)))

def night_to_day(image):
	sharpen_filter = np.array([ [0, -1, 0], [-1, 5, -1], [0, -1, 0]])
	working = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	contrast = cv2.cvtScaleAbs(working, alpha=1.4)
	sharp = cv2.filter2D(contrast, -1, sharpen_filter)
	sharp = cv2.filter2D(sharp, -1, sharpen_filter)
	return imagefilters.white_balance(sharp)

# https://www.scitepress.org/Papers/2021/102385/102385.pdf
def extract_drivable(input_image, is_night):
    if is_night:
        corrected_image = adjust_gamma(night_to_day(input_image), 2.2)
        return corrected_image
    
    corrected_image = adjust_gamma(input_image, 2.2)
    c1c2c3_image = convert_bgr2c1c2c3(corrected_image)
    hsv_image = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2HSV)
    



