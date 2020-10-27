import numpy as np
import cv2
import torch

from SINet import *

# Load the sinet pytorch model
config = [[[3, 1], [5, 1]], [[3, 1], [3, 1]],
              [[3, 1], [5, 1]], [[3, 1], [3, 1]], [[5, 1], [3, 2]], [[5, 2], [3, 4]],
              [[3, 1], [3, 1]], [[5, 1], [5, 1]], [[3, 2], [3, 4]], [[3, 1], [5, 2]]]
model = SINet(classes=2, p=2, q=8, config=config,chnn=1)
model.load_state_dict(torch.load('model_296.pth'))
model.eval()

# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load and resize the background image
bgd = cv2.imread('whitehouse.jpeg')
bgd = cv2.resize(bgd, (320,320))
bgd = bgd.astype(np.float32)

# Mean and std. deviation for normalization
mean = [102.890434, 111.25247,  126.91212 ]
std = [62.93292,  62.82138,  66.355705]
h,w = 320, 320

# Capture video from camera
cap = cv2.VideoCapture(0)

while cv2.waitKey(1) < 0:

    # Read input frames
    success, frame = cap.read()
    if not success:
           vidcap.release()
           break

    # Resize the input video frame
    frame = cv2.resize(frame, (h,w))
    frame = frame.astype(np.float32)
    
    # Normalize and add batch dimension
    img = frame
    img = (img-mean)/std
    img /= 255
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis,...]

    # Load the inputs into GPU   
    inputs=torch.from_numpy(img).float().to(device)

    # Perform prediction and plot results
    with torch.no_grad():    
         torch_res = model(inputs)
         _, mask = torch.max(torch_res, 1)
      
    # Alpha blending with background image
    mask = mask.view(h,w,1).cpu().numpy()
    blend = (frame * mask) + (bgd * (1-mask))

    # Show output in window
    cv2.imshow('Portrait Segmentation', np.uint8(blend))

'''
Run: python3 sinet_video.py
Issues: 'RuntimeError: max_pool2d_with_indices_out_cuda_frame failed with error code 0' in GPU on older pytorch versions(1.4)
'''
