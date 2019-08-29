
from PIL import Image
import os, sys
import cv2
import numpy as np

'''
Images: SxS 
Values: 0-255
Format: (PNG or JPEG) RGB
'''

# Convert source images into '.npy' format 

img_path = "/path/to/image/dir/"
img_dirs = os.listdir( img_path )
img_dirs.sort()
x_train=[]

def load_image():
    for item in img_dirs:
        if os.path.isfile(img_path+item):
            im = Image.open(img_path+item).convert("RGB")
            im = np.array(im)
            im=im.astype('uint8')
            x_train.append(im)
load_image()

imgset=np.array(x_train)
np.save("img_uint8.npy",imgset)


'''
Masks: SxS 
Values: 0 (background) and 255 (foreground)
Format: PNG (RGB or ALPHA) 
'''

# Convert mask images into '.npy' format

msk_path = "/path/to/mask/dir/"
msk_dirs = os.listdir( msk_path )
msk_dirs.sort()
y_train=[]

def load_mask():
    for item in msk_dirs:
        if os.path.isfile(msk_path+item):
            im = Image.open(msk_path+item).convert("RGB")
            im = np.array(im)
            im=im[...,0].astype('uint8')
            im=im.astype('uint8')
            y_train.append(im)
load_mask()

mskset=np.array(y_train)
np.save("msk_uint8.npy",mskset[...,np.newaxis])
