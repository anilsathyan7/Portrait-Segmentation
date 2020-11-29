import sys, os
import dlib
import cv2

'''
Crops portrait images and masks from person segmentation datasets.
'''

# Load the cnn face detector
detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

# Path to source images and masks
img_path = 'img/'
msk_path = 'masks_machine/'

# Path for saving cropped outputs
crop_img_path = 'crop_images/'
crop_msk_path = 'crop_masks/'

# Get the image file names
dirs_img = os.listdir(img_path)
dirs_msk = os.listdir(msk_path)

# Set minimum height and width
min_crop_height, min_crop_width = 128, 128

avg_height, avg_width = 2048, 2048
max_height, max_width = 6144, 6144

# Set target height and width
target_height, target_width = 256, 256

for item in dirs_img:
     if os.path.isfile(img_path+item):

        print("Image path: ",img_path+item)

        # Ensure masks are in png format
        png_msk=item.rsplit('.',1)[0]+'.png' 
    
        # Read the input image
        img = cv2.imread(img_path+item) 
        msk = cv2.imread(msk_path+png_msk)

        # Get the image shape
        rows, cols, _ = img.shape
  
        # Count number of faces
        fcount=0       

        # Resize images if input size exceeds maximum limit
        if img.shape[0] > max_height  or img.shape[1]> max_width :
            img =  cv2.resize(img,(max_height, max_width))
            msk =  cv2.resize(msk, (max_height, max_width),interpolation = cv2.INTER_NEAREST)
        
        print('Image shape: ', img.shape)

        # Perform inference on image
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if rgb_image.shape[0] < avg_height  or rgb_image.shape[1] < avg_width :
           dets = detector(rgb_image,1)
         
        else: 
           dets = detector(rgb_image) # Do not upsample      

   
        # Crop and save the portrait face images, along with mask
        for det in dets:
   
            width, height = det.rect.right() - det.rect.left(), det.rect.bottom() - det.rect.top()

            y0 = max(0,det.rect.top()-width)
            y1 = min(det.rect.bottom()+width, rows)
            x0 = max(0,det.rect.left()-width)
            x1 = min(det.rect.right()+width, cols)
            w = (x1-x0)
            h = (y1-y0)
           
            # Crop the face ROI
            img_crop = img[y0:y0+h , x0:x0+w]
            msk_crop = msk[y0:y0+h , x0:x0+w]
           
            print("Crop size: ", img_crop.shape)

            if img_crop.shape[0] > min_crop_height or img_crop.shape[1] > min_crop_width :
              if len(detector(img_crop))==1: # Check if there are multiple faces
                cv2.imwrite(crop_img_path+"port_"+str(fcount)+item, cv2.resize(img_crop,(target_height, target_width)))
                cv2.imwrite(crop_msk_path+"port_"+str(fcount)+png_msk, cv2.resize(msk_crop, (target_height, target_width),interpolation = cv2.INTER_NEAREST))
                fcount=fcount+1
'''
Configure the directory paths and run: python3 face_crop.py

Modify the max and average size depending upon the GPU memory.
Manually verfiy all the images to see if it contains multiple faces. 
'''
