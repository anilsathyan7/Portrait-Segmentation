import os, sys
from imageio import imsave
import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline

'''
A script for applying curve filters on dataset.
Configure the source and save paths, and 
run the script: python curve.py 
'''

# Configure the filepaths
path = "/path/to/dataset/source/"
save_warm = "/path/to/dataset/warm/"
save_cool = "/path/to/dataset/cool/"
dirs = os.listdir( path )

# Interpolation look-up-table
def _create_LUT_8UC1(x, y):
   spl = UnivariateSpline(x, y)
   return spl(range(256))

# Curve filter class
class CurveFilter:
 
  def __init__(self):
    self.incr_ch_lut = _create_LUT_8UC1([0, 64, 128, 192, 256],[0, 70, 140, 210, 256])
    self.decr_ch_lut = _create_LUT_8UC1([0, 64, 128, 192, 256],[0, 30,  80, 120, 192])


  # Warm filter
  def warm(self, img_rgb):
    c_r, c_g, c_b = cv2.split(img_rgb)
    c_r = cv2.LUT(c_r, self.incr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, self.decr_ch_lut).astype(np.uint8)
    img_rgb = cv2.merge((c_r, c_g, c_b))
    
    # Increase color saturation
    c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
    c_s = cv2.LUT(c_s, self.incr_ch_lut).astype(np.uint8)
    img = cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)
    return img

  # Cool filter
  def cool(self, img_rgb):

     c_r, c_g, c_b = cv2.split(img_rgb)
     c_r = cv2.LUT(c_r, self.decr_ch_lut).astype(np.uint8)
     c_b = cv2.LUT(c_b, self.incr_ch_lut).astype(np.uint8)
     img_rgb = cv2.merge((c_r, c_g, c_b))

     # Decrease color saturation
     c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
     c_s = cv2.LUT(c_s, self.decr_ch_lut).astype(np.uint8)
     img = cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)
     return img

# Create a curve filter object
cf=CurveFilter()

# Apply filters to dataset
def curve():
    
    for item in dirs:
        if os.path.isfile(path+item):
            # Read input image
            img = cv2.imread(path+item)
            img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply the filters
            cool=cf.cool(img)
            warm=cf.warm(img)

            # Save output images
            imsave(save_warm+item,warm)
            imsave(save_cool+item,cool)
         
curve()
