# Portrait-Segmentation

A **Real-time Automatic Deep Matting** approach for **mobile devices**, using **Mobile-Unet and Keras**.

**Portrait segmentation** refers to the process of segmenting a person in an image from its background.
Here we use the concept of **semantic segmentation** to predict the label of every pixel(dense prediction) in an image.

Here we limit ourselves to **binary classes** (person or background) and use only plain **portrait-selfie** images for matting.

## Dependencies

* Tensorflow(>=1.14.0), Python 3
* Keras(>=2.2.4), Kito
* Opencv, PIL, Matplotlib

```
pip uninstall -y tensorflow
pip install -U tf-nightly
pip install keras
pip install kito
```

## Prerequisites

* Download training [data-set](https://drive.google.com/file/d/1UBLzvcqvt_fin9Y-48I_-lWQYfYpt_6J/view?usp=sharing)
* GPU with CUDA support

## Dataset

The dataset consists of **18698 human portrait images of size 128x128 in RGB format**, along with their**masks(ALPHA)**. Here we augment the **PFCN** dataset with (handpicked) portrait  images form **supervisely** dataset.Additionaly,we download **random selfie** images from web and generate their masks using state-of-the-art **deeplab-xception** network. 

Now to increase the size of dataset, we perform augmentation like **cropping, brightness alteration and flipping**. Since most of our images contain plain background, we create new **synthetic images** using random backgrounds (natural) using the default dataset, with the help of a **python script**.

Besides the aforesaid augmentation techniques, we **normalize(also standardize)** the images and perform **run-time augmentations like flip, shift and zoom** using keras data generator and preprocessing module.

## Model Architecture

Here we use **Mobilent v2** with **depth multiplier 0.5** as encoder (feature extractor).

For the **decoder part**, we have two variants. You can use a upsampling block with either  **Transpose Convolution** or **Upsample2D+Convolution**. In the former case we use a **stride of 2**, whereas in the later we use **resize bilinear** for upsampling, along with Conv2d. Ensure proper **skip connections** between encoder and decoder parts for better results.

Here is the snapshot of the upsampled version of model.

![Screenshot](portrait_seg_small.png)

## How to run

Download the dataset from the above link and put them in data folder.
After ensuring the data files are stored in the desired directorires, run the scripts in the following order.

```python
1. python train.py # Train the model on data-set
2. python eval.py checkpoints/up_super_model-102-0.06.hdf5 # Evaluate the model on test-set
3. python export.py checkpoints/up_super_model-102-0.06.hdf5 # Export the model for deployment
4. python test.py test/four.jpeg # Test the model on a single image
5. python webcam.py test/beach.jpg # Run the model on webcam feed
```

## Demo

### Result

![Screenshot](result.png)

### Android Application

## Key Insights and Drawbacks

1. Always start experimentation with standard/pretrained networks. Also try out default/standard hyperparameter settings before experimentation.
2. Make sure you ground truth is correct/uncorrupted and is in desired format before training (even standard dataset).
3. For mobile devices, make sure you use a mobile-friendly architecture (like mobilenet) for training and deployment.
4. Using google colaboratory along with google drive for taining was EASY & FUN.It provides high end GPU (RAM also) for free.
5. Some of the mobile optimization tools(even TF) are still experimental (GPU deegate,NNAPI,FP16 etc.) and are buggy.They support only limited operations and edge devices.

## TODO

* Port the code to TF 2.0
* Use a bigger image size (224x224)
* Try quantization-aware training
* Train with mixed precision (FP16)
* Improve accuracy & reduce artifacts at runtime

## Versioning

Version 1.0

## Authors

Anil Sathyan

## Acknowledgments
* https://www.tensorflow.org/model_optimization
* https://github.com/cainxx/image-segmenter-ios
* https://github.com/gallifilo/final-year-project
* https://github.com/lizhengwei1992/mobile_phone_human_matting
*   [Deeplab Image Segmentation](https://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb)
*   [Tensorflow - Image segmentation](https://www.tensorflow.org/beta/tutorials/images/segmentation)
*   [Hyperconnect - Tips for fast portrait segmentation](https://hyperconnect.github.io/2018/07/06/tips-for-building-fast-portrait-segmentation-network-with-tensorflow-lite.html)
* [Keras Documentation](https://keras.io/)
* [Boundary-Aware Network for Fast and High-Accuracy Portrait Segmentation](https://arxiv.org/pdf/1901.03814.pdf)
* [Fast Deep Matting for Portrait Animation on Mobile Phone](https://arxiv.org/pdf/1707.08289.pdf)
