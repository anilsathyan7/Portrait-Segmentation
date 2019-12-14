import numpy as np
import cv2 
from  scipy import ndimage

'''
A script for foreground segmenation using background subtraction.
The algorithms assumes a static background scenary.The foreground 
object should be introduced into the scene only after window pop-up.
'''

# Create a video capturer
cap = cv2.VideoCapture(0)

# Read first 100 frames and find mean image
_, first_frame = cap.read()
mean_bgd = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

for i in range(0,99):
    _, first_frame = cap.read()
    frames=cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    frames=np.int32(frames)
    mean_bgd=mean_bgd+frames
mean_bgd=np.uint8(mean_bgd/100)

# Apply gaussian blur to remove noise
mean_bgd = cv2.GaussianBlur(mean_bgd, (5, 5), 0)

print("Inference started...")

# Create a plain background image
plain_bgd = np.zeros(shape=(mean_bgd.shape[0],mean_bgd.shape[1],3),dtype=np.uint8)
plain_bgd[:] = (0, 0, 0)

while True:

    # Read frames from webcam and convert it to grayscale
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # Find differece between current frame and mean image
    difference = cv2.absdiff(mean_bgd, gray_frame)
    _, fgMask = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)

    # Fill up the gaps using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(53,53))
    fgMask= cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
    fgMask=ndimage.binary_fill_holes(fgMask).astype(np.uint8)

    # Find the largest contour and remove smaller blobs
    _, contours,hierarchy = cv2.findContours(fgMask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    finmsk=np.zeros_like(fgMask)
    if areas:
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        cv2.fillPoly(finmsk, pts =[cnt], color=(255,255,255))
    
    # Apply gaussian blur on mask  
    finmsk=cv2.GaussianBlur(finmsk, (25, 25), 0)
    finmsk=finmsk[...,np.newaxis]/255
    
    # Alpha blend frame with background, using the mask
    result=np.uint8(finmsk*frame + (1.0 -finmsk)*plain_bgd)

    # Display output frame
    cv2.imshow("Output", result)

    # Exit on keyboard interrupt
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
print("Keyboard interrupt...")
cap.release()
cv2.destroyAllWindows()
