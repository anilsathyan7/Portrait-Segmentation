# Import libraries
import os
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Input,Flatten, concatenate,Reshape, Conv2D, MaxPooling2D, Lambda,Activation,Conv2DTranspose
from keras.layers import UpSampling2D, Conv2DTranspose, BatchNormalization, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback, ReduceLROnPlateau
from keras.optimizers import SGD, Adam
import keras.backend as K
from keras.utils import plot_model
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from random import randint
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
from random import randint
# %matplotlib inline

# Load the dataset
x_train=np.load("data/img_uint8.npy")
y_train=np.load("data/msk_uint8.npy")

# Configure save paths and batch size
PRETRAINED='checkpoints/pretrained_model.hdf5'
CHECKPOINT="checkpoints/bilinear_segmodel-{epoch:02d}-{val_loss:.2f}.hdf5"
LOGS='./logs'
BATCH_SIZE=32

# Verify the mask shape and values
print(np.unique(y_train))
print(y_train.shape)

# Total number of images
num_images=x_train.shape[0]

# Preprocessing function (runtime)
def normalize_batch(imgs):
    if imgs.shape[-1] > 1 :
      return (imgs -  np.array([0.50693673, 0.47721124, 0.44640532])) /np.array([0.28926975, 0.27801928, 0.28596011])
    else:
      return imgs.round()
def denormalize_batch(imgs,should_clip=True):
    imgs= (imgs * np.array([0.28926975, 0.27801928, 0.28596011])) + np.array([0.50693673, 0.47721124, 0.44640532])
    
    if should_clip:
        imgs= np.clip(imgs,0,1)
    return imgs


# Data generator for training and validation

data_gen_args = dict(rescale=1./255,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     validation_split=0.2
                    )

image_datagen = ImageDataGenerator(**data_gen_args, preprocessing_function=normalize_batch)
mask_datagen = ImageDataGenerator(**data_gen_args,  preprocessing_function=normalize_batch)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
batch_sz=BATCH_SIZE

# Train-val split (80-20)
num_train=int(num_images*0.8)
num_val=int(num_images*0.2) 

train_image_generator = image_datagen.flow(
    x_train,
    batch_size=batch_sz,
    shuffle=True,
    subset='training',
    seed=seed)

train_mask_generator = mask_datagen.flow(
    y_train,
    batch_size=batch_sz,
    shuffle=True,
    subset='training',
    seed=seed)


val_image_generator = image_datagen.flow(
    x_train, 
batch_size = batch_sz,
shuffle=True,
subset='validation',
seed=seed)

val_mask_generator = mask_datagen.flow(
     y_train,
batch_size = batch_sz,
shuffle=True,
subset='validation',
seed=seed)

                     
# combine generators into one which yields image and masks

train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)


# Convolution block with Transpose Convolution
def deconv_block(tensor, nfilters, size=3, padding='same', kernel_initializer = 'he_normal'):
    
    y = Conv2DTranspose(filters=nfilters, kernel_size=size, strides=2, padding = padding, kernel_initializer = kernel_initializer)(tensor)
    y = BatchNormalization()(y)
    y = Dropout(0.5)(y)
    y = Activation("relu")(y)
    
    
    return y

# Convolution block with Upsampling+Conv2D
def deconv_block_rez(tensor, nfilters, size=3, padding='same', kernel_initializer = 'he_normal'):
    y = UpSampling2D(size = (2,2),interpolation='bilinear')(tensor)
    y = Conv2D(filters=nfilters, kernel_size=(size,size), padding = 'same', kernel_initializer = kernel_initializer)(y)
    y = BatchNormalization()(y)
    y = Dropout(0.5)(y)
    y = Activation("relu")(y)
    
    
    return y

# Model architecture
def get_mobile_unet(finetune=False, pretrained=False):

    # Load pretrained model (if any)
    if (pretrained):
       model=load_model(PRETRAINED)
       print("Loaded pretrained model ...\n")
       return model
    
    # Encoder/Feature extractor
    mnv2=keras.applications.mobilenet_v2.MobileNetV2(input_shape=(128, 128, 3),alpha=0.5, include_top=False, weights='imagenet')
    
    if (finetune):
      print("Freezing initial layer ...\n")
      for layer in mnv2.layers[:-3]:
        layer.trainable = False
    
    x = mnv2.layers[-4].output

    # Decoder
    x = deconv_block_rez(x, 512)
    x = concatenate([x, mnv2.get_layer('block_13_expand_relu').output], axis = 3)
    
    x = deconv_block_rez(x, 256)
    x = concatenate([x, mnv2.get_layer('block_6_expand_relu').output], axis = 3)
                
    x = deconv_block_rez(x, 128)
    x = concatenate([x, mnv2.get_layer('block_3_expand_relu').output], axis = 3)
    
    x = deconv_block_rez(x, 64)
    x = concatenate([x, mnv2.get_layer('block_1_expand_relu').output], axis = 3)
                

    #x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', kernel_initializer = 'he_normal')(x)
    x = UpSampling2D(size = (2,2),interpolation='bilinear')(x)
    x = Conv2D(filters=32, kernel_size=3, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
   
    x = Conv2DTranspose(1, (1,1), padding='same')(x)
    x = Activation('sigmoid', name="op")(x)
    
    
    model = Model(inputs=mnv2.input, outputs=x)
    
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3),metrics=['accuracy'])
    return model
  
model=get_mobile_unet()

# Model summary
model.summary()

# Plot model architecture
plot_model(model, to_file='portrait_seg.png')

# Save checkpoints
checkpoint = ModelCheckpoint(CHECKPOINT, monitor='val_loss', verbose=1, save_weights_only=False , save_best_only=True, mode='min')

# Callbacks 
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=15, min_lr=0.000001, verbose=1)
tensorboard = TensorBoard(log_dir=LOGS, histogram_freq=0,
                          write_graph=True, write_images=True)

callbacks_list = [checkpoint, tensorboard,reduce_lr]



# Train the model
model.fit_generator(
    train_generator,
    epochs=300,
    steps_per_epoch=num_train/batch_sz,
    validation_data=val_generator, 
    validation_steps=num_val/batch_sz,
    use_multiprocessing=True,
    workers=2,
    callbacks=callbacks_list)

#Sample run: python train.py
