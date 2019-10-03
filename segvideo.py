import numpy as np
import cv2
import sys
import time
import keras
from keras.models import Model
from keras.models import load_model

def image_stats(image):

    # Compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # Return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)
 
def color_transfer(source, target):
    
    # Convert images to UINT8 (0-255)
    source = np.uint8(source*255.0)
    target = np.uint8(target*255.0)

    # Convert the images from the RGB to L*ab* color space
    source = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype("float32")

    # Compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)
 
    # Subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar
 
    # Scale by the standard deviations
    l = (lStdTar / lStdSrc) * l
    a = (aStdTar / aStdSrc) * a
    b = (bStdTar / bStdSrc) * b
 
    # Add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # Clip the pixel intensities to [0, 255]
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)
 
    # Merge the channels together and convert back to the RGB format
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2RGB)
	
    # Convert image to float (0-1)
    transfer=transfer/255.0   

    # Return the color transferred image
    return transfer

def smoothstep(edge0, edge1, x) :
    # Scale, bias and saturate x to 0..1 range
    x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    #Evaluate polynomial
    return x * x * (3 - 2 * x)


def seamlessclone(source, mask):
   
    # Convert images to UINT8 (0-255)
    src=np.uint8(source*255.0)
    dst = np.uint8(bgd*255.0)
    msk=np.uint8(mask*255.0)

    # Dilate the mask
    kernel = np.ones((7,7),np.uint8)
    msk = cv2.dilate(msk,kernel,iterations = 1)

 
    # Convert images to BGR format
    src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
    dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)

    # Clone size
    clone_size=tgt_size-2

    # Resize images
    src = cv2.resize(src, (clone_size,clone_size),interpolation = cv2.INTER_LINEAR)
    msk = cv2.resize(msk, (clone_size,clone_size),interpolation = cv2.INTER_LINEAR)

    # Find contours of mask ROI
    contours, hierarchy = cv2.findContours(msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key = cv2.contourArea)   

    # Find ROI co-ordinates
    (x,y,w,h) = cv2.boundingRect(largest)
    X = x+w//2
    Y = clone_size-h//2
    
    # Get ROI center
    center = (X,Y)
    #print(X+w//2,Y+h//2)

    # Seamless cloning
    clone = cv2.seamlessClone(src, dst, msk, center, cv2.NORMAL_CLONE)
    clone = cv2.cvtColor(clone, cv2.COLOR_BGR2RGB)
  
    return clone

def change_bgd(x):
    # Select background image
    global bgd
    if x == 0:
       bgd = cv2.resize(cv2.imread('test/desert.jpg'), (tgt_size,tgt_size))
       bgd = cv2.cvtColor(bgd, cv2.COLOR_BGR2RGB)/255.0
    elif x == 1:
       bgd = cv2.resize(cv2.imread('test/ocean.jpeg'), (tgt_size,tgt_size))
       bgd = cv2.cvtColor(bgd, cv2.COLOR_BGR2RGB)/255.0
    elif x == 2:
       bgd = cv2.resize(cv2.imread('test/sky.jpg'), (tgt_size,tgt_size))
       bgd = cv2.cvtColor(bgd, cv2.COLOR_BGR2RGB)/255.0
    elif x == 3:
       bgd = cv2.resize(cv2.imread('test/sunset.jpg'), (tgt_size,tgt_size))
       bgd = cv2.cvtColor(bgd, cv2.COLOR_BGR2RGB)/255.0
    else:
       bgd = cv2.resize(cv2.imread('test/blue.jpg'), (tgt_size,tgt_size))
       bgd = cv2.cvtColor(bgd, cv2.COLOR_BGR2RGB)/255.0

def harmonize(image, mask):

    # Convert image to BGR format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Resize the images
    image=np.float32( cv2.resize(image,(512,512)) )
    mask=np.float32( cv2.resize(mask, (512,512)) ) - 128.0
    
    # Generate blob inputs from images
    blobimg = cv2.dnn.blobFromImage(image, 1, (512, 512), (104.00699, 116.66877, 122.67892))
    blobmsk = cv2.dnn.blobFromImage(mask, 1, (512, 512))

    # Feed the inputs
    net.setInput(blobimg,'data')
    net.setInput(blobmsk,'mask')

    # Predict the output
    start = time.time()
    pred = net.forward()
    end = time.time()
    print('Time: '+str(end-start))
 
    # Add mean to output
    res = pred[0].transpose((1,2,0))
    res += np.array((104.00699, 116.66877, 122.67892))
    res = res[:,:,::-1]
  
    # Clip pixel values
    res=np.clip(res,0.0,255.0)

    # Resize the output image   
    img = res.astype(np.uint8)
    img = cv2.resize(img,(tgt_size,tgt_size))

    return img

   
# Load the model and background image
model = load_model('models/transpose_seg/deconv_bnoptimized_munet.h5', compile=False)

# Load the caffe model for colour harmonization
prototxt='models/caffe/deploy_512.prototxt'
weights='models/caffe/harmonize_iter_200000_fp16.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, weights)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

# Target size
tgt_size=300

# Convert background to float [0-1]
bgd = cv2.resize(cv2.imread(sys.argv[1]), (tgt_size,tgt_size))
bgd = cv2.cvtColor(bgd, cv2.COLOR_BGR2RGB)/255.0

# Initialize video capturer
cap = cv2.VideoCapture(0)

# Create a named window
cv2.namedWindow('portrait segmentation')

# Create trackbars for background selection
cv2.createTrackbar('BGD','portrait segmentation',0,4,change_bgd)


while(True):
     
    # Get keyboard input
    key = cv2.waitKey(2) & 0xFF
    if key == ord('c'):
        filter='color_transfer'
    elif key == ord('s'):
        filter='seamless_clone' 
    elif key == ord('m'):
        filter='smooth_step'
    elif key == ord('h'):
        filter='colour_harmonize'

    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret: 
      # Pre-process
      img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      simg = cv2.resize(img,(128,128),interpolation=cv2.INTER_AREA)
      simg = simg.reshape((1,128,128,3))/255.0
      
      # Predict
      out=model.predict(simg)
      orimsk=np.float32((out>0.5)).reshape((128,128,1))

      # Post-process
      msk=cv2.GaussianBlur(orimsk,(7,7),1)
      img=cv2.resize(img, (tgt_size,tgt_size))/255.0
      msk=cv2.resize(msk, (tgt_size,tgt_size)).reshape((tgt_size,tgt_size,1))
      

      # Colour transfer
      if filter=='color_transfer':
        img=color_transfer(bgd,img)

      # Smooth step
      elif filter=='smooth_step':
        msk=smoothstep(0.3, 0.5,msk)

      # Seamless clone
      elif filter=='seamless_clone':
        frame=seamlessclone(img, orimsk)
      
      # Alpha blending
      if filter!='seamless_clone' :
        frame = (img * msk) + (bgd * (1 - msk))
        frame = np.uint8(frame*255.0)
        mask = np.uint8(msk*255.0)
         
      if filter=='colour_harmonize' :
        frame=harmonize(frame,mask)

      # Display the resulting frame
      cv2.imshow('portrait segmentation',frame[...,::-1])
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capturer
cap.release()
cv2.destroyAllWindows()

# Sample run: python segvideo.py test/sunset.jpg
