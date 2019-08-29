import numpy as np
from PIL import Image
import tensorflow as tf
import keras
from keras.models import Model
import sys
from keras.models import load_model
from keras.layers import Input, Flatten
import matplotlib.pyplot as plt
from kito import reduce_keras_model # Ensure kito is installed

# Load the model (output of training - checkpoint)
model=load_model(sys.argv[1])

# Fold batch norms
model_reduced = reduce_keras_model(model)
model_reduced.save('bilinear_bnoptimized_munet.h5') # Use this model in PC

# Flatten output and save model (Optimize for phone)
output = model_reduced.output
newout=Flatten()(output)
new_model=Model(model_reduced.input,newout)

new_model.save('bilinear_fin_munet.h5')


# For Float32 Model
converter = tf.lite.TFLiteConverter.from_keras_model_file('bilinear_fin_munet.h5')
tflite_model = converter.convert()
open("bilinear_fin_munet.tflite", "wb").write(tflite_model)


#For UINT8 Quantization
converter = tf.lite.TFLiteConverter.from_keras_model_file('bilinear_fin_munet.h5')
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE] 
converter.post_training_quantize=True
tflite_model = converter.convert()
open("bilinear_fin_munet_uint8.tflite", "wb").write(tflite_model)


#For Float16 Quantization (Requires TF 1.15 or above)
converter = tf.lite.TFLiteConverter.from_keras_model_file('bilinear_fin_munet.h5')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
tflite_model = converter.convert()
open("bilinear_fin_munet_fp16.tflite", "wb").write(tflite_model)


# Sample run: python export.py checkpoints/up_super_model-102-0.06.hdf5

