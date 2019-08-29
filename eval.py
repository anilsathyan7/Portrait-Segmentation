import numpy as np
import keras
import sys
from keras.models import load_model



# Load a trained model checkpoint
model=load_model(sys.argv[1])

# Load a test dataset (UINT8)
new_xtest=np.load('data/test_xtrain.npy')
new_ytest=np.load('data/test_ytrain.npy')

# Evaluate model 
score = model.evaluate(np.float32(new_xtest/255.0), np.float32(new_ytest/255.0), verbose=0)

# Print loss and accuracy
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Sample run: python eval.py checkpoints/up_super_model-102-0.06.hdf5
