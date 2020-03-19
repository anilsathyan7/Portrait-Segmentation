import cv2
import numpy as np
import tensorflow as tf
import time

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.framework import convert_to_constants
from tensorflow.keras.models import Model, load_model

# Convert keras model to saved model format
model = load_model('models/prisma_seg/prisma-net-15-0.08.hdf5', compile=False)
model.save('saved_model')

# Configure the input and output paths
input_saved_model_dir='saved_model'
output_saved_model_dir='tftrt_model'

# Configure the conversion parameters
params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
         precision_mode='FP16',
         minimum_segment_size=7,
         max_workspace_size_bytes=1 << 32,
         maximum_cached_engines=100)
converter = trt.TrtGraphConverterV2(
         input_saved_model_dir, conversion_params=params)
converter.convert()

# Optimize the converted function with inputs
num_runs=100
def my_input_fn():
  for _ in range(num_runs):
    Inp1 = np.random.normal(size=(1, 256, 256, 3)).astype(np.float32)
    yield [Inp1]

# Convert and save the output model
converter.build(input_fn=my_input_fn)
converter.save(output_saved_model_dir)

# Sample run: python3 tftrt_convert.py
