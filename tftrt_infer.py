import numpy as np
import cv2
import sys
import time
import tensorflow as tf

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.framework import convert_to_constants

# Load the tftrt model 
tftrt_model_dir='tftrt_model'
saved_model_loaded = tf.saved_model.load(
    tftrt_model_dir, tags=[tag_constants.SERVING])
graph_func = saved_model_loaded.signatures[
    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
frozen_func = convert_to_constants.convert_variables_to_constants_v2(
    graph_func)

# Load background image
bgd = cv2.resize(cv2.imread(sys.argv[1]), (513,513))
bgd = cv2.cvtColor(bgd, cv2.COLOR_BGR2RGB)/255.0

# Capture video from webcam
cap = cv2.VideoCapture(0) 

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret: 
      # Preprocess
      img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      np_img = cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA)
      np_img = np_img.reshape((1,256,256,3))/255.0
      tr_img=tf.convert_to_tensor(np_img, dtype=tf.float32)
      
      # Predict
      start = time.perf_counter()
      out=frozen_func(tr_img)[0].numpy()
      ex_time=time.perf_counter()-start
      print("Inference time: {:04.2f} ms".format(ex_time*1000))
      msk=np.float32((out>0.5)).reshape((256,256,1))

      # Post-process
      msk=cv2.GaussianBlur(msk,(5,5),1)
      img=cv2.resize(img, (513,513))/255.0
      msk=cv2.resize(msk, (513,513)).reshape((513,513,1))
      
      # Alpha blending
      frame = (img * msk) + (bgd * (1 - msk))
      frame = np.uint8(frame*255.0)

      # Display the resulting frame
      cv2.imshow('portrait segmentation',frame[...,::-1])
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# Sample run: python3 tftrt_infer.py test/beach.jpg
