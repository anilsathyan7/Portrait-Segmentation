import cv2, sys, time
import numpy as np
import tensorflow as tf
from PIL import Image

# Width and height
width  = 320
height = 240

# Frame rate
fps = ""
elapsedTime = 0

# Video capturer
cap = cv2.VideoCapture(0)
cv2.namedWindow('FPS', cv2.WINDOW_AUTOSIZE)

# Initialize tflite-interpreter
interpreter = tf.lite.Interpreter(model_path="models/transpose_seg/deconv_fin_munet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]

# Image overlay
overlay = np.zeros((input_shape[0],input_shape[1], 3), np.uint8)
overlay[:] = (127, 0, 0)

while True:
     
    # Read frames
    t1 = time.time()
    ret, frame = cap.read()

    # BGR->RGB, CV2->PIL
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image=Image.fromarray(rgb)
    
    # Resize image
    image= image.resize(input_shape, Image.ANTIALIAS)

    # Normalization
    image = np.asarray(image)
    prepimg = image / 255.0
    prepimg = prepimg[np.newaxis, :, :, :]

    # Segmentation
    interpreter.set_tensor(input_details[0]['index'], np.array(prepimg, dtype=np.float32))
    interpreter.invoke()
    outputs = interpreter.get_tensor(output_details[0]['index'])

    # Process the output
    output = np.uint8(outputs[0]>0.5)
    res=np.reshape(output,input_shape)
    mask = Image.fromarray(np.uint8(res), mode="P")
    mask = np.array(mask.convert("RGB"))*overlay
    mask = cv2.resize(np.asarray(mask), (width,height),interpolation=cv2.INTER_CUBIC)
    frame = cv2.resize(frame, (width,height),interpolation=cv2.INTER_CUBIC)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    
    # Overlay the mask
    output = cv2.addWeighted(frame, 1, mask, 0.9, 0)
    
    # Display the output
    cv2.putText(output, fps, (width-180,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
    cv2.imshow('FPS', output)

    if cv2.waitKey(1)&0xFF == ord('q'):
        break
    elapsedTime = time.time() - t1

    # Print frame rate
    fps = "(Playback) {:.1f} FPS".format(1/elapsedTime)
    print("fps = ", str(fps))

cap.release()
cv2.destroyAllWindows()

