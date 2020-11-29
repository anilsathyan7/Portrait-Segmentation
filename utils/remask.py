import os, sys, imageio, glob
from PIL import Image
import numpy as np

# Configure input and output directory paths
source_mask_list = glob.glob('/path/to/source/Labels/*.*', recursive=True)
target_mask_path = "/path/to/target/Labels/"

# Set foreground pixel value and target image size
fgd_pixval = 255
tgt_size = (256,256)

print(source_mask_list)

# Convert and save the binary masks
def remask():

    image_count=0

    for item in source_mask_list:
        if os.path.isfile(item):

            # Convert the mask to binary alpha
            im = Image.open(item)
            im = im.split()[-1]
            im = im.resize(tgt_size,Image.NEAREST)
            
            # Threshold the image using value 127
            im = np.array(im)
            im[im>=127]=fgd_pixval
            im[im<127]=0 
           
            # Save the alpha mask
            file_name=os.path.basename(item)
            imageio.imsave(target_mask_path+file_name,im)
            image_count=image_count+1
            print("Processing: "+str(image_count))

if __name__ == "__main__":             
   remask()
