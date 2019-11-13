# Portrait-Segmentation

**Real-time Automatic Deep Matting For Mobile Devices With Mobile-Unet**

**Portrait segmentation** refers to the process of segmenting a person in an image from its background.
Here we use the concept of **semantic segmentation** to predict the label of every pixel (dense prediction) in an image.

Here we limit ourselves to **binary classes** (person or background) and use only plain **portrait-selfie** images for matting.

## Dependencies

* Tensorflow(>=1.14.0), Python 3
* Keras(>=2.2.4), Kito
* Opencv(>=3.4), PIL, Matplotlib

```
pip uninstall -y tensorflow
pip install -U tf-nightly
pip install keras
pip install kito
```

## Prerequisites

* Download training [data-set](https://drive.google.com/file/d/1UBLzvcqvt_fin9Y-48I_-lWQYfYpt_6J/view?usp=sharing)
* GPU with CUDA support
* Download caffe harmonization [model](https://drive.google.com/file/d/1bWafRdYBupr8eEuxSclIQpF7DaC_2MEY/view?usp=sharing)

## Dataset

The dataset consists of **18698 human portrait images of size 128x128 in RGB format**, along with their **masks(ALPHA)**. Here we augment the [**PFCN**](https://1drv.ms/u/s!ApwdOxIIFBH19Ts5EuFd9gVJrKTo) dataset with (handpicked) portrait  images form **supervisely** dataset. Additionaly, we download **random selfie** images from web and generate their masks using state-of-the-art **deeplab-xception** model for semantic segmentation. 

Now to increase the size of dataset, we perform augmentation like **cropping, brightness alteration and flipping**. Since most of our images contain plain background, we create new **synthetic images** using random backgrounds (natural) using the default dataset, with the help of a **python script**.

Besides the aforesaid augmentation techniques, we **normalize(also standardize)** the images and perform **run-time augmentations like flip, shift and zoom** using keras data generator and preprocessing module.

## Model Architecture

Here we use **Mobilent v2** with **depth multiplier 0.5** as encoder (feature extractor).

For the **decoder part**, we have two variants. You can use a upsampling block with either  **Transpose Convolution** or **Upsample2D+Convolution**. In the former case we use a **stride of 2**, whereas in the later we use **resize bilinear** for upsampling, along with Conv2d. Ensure proper **skip connections** between encoder and decoder parts for better results.

Additionaly, we use **dropout** regularization to prevent **overfitting**.It also helps our network to learn more **robust** features during training.

Here is the **snapshot** of the **upsampled** version of model.

![Screenshot](portrait_seg_small.png)


## How to run

Download the **dataset** from the above link and put them in **data** folder.
After ensuring the data files are stored in the **desired directorires**, run the scripts in the **following order**.

```python
1. python train.py # Train the model on data-set
2. python eval.py checkpoints/up_super_model-102-0.06.hdf5 # Evaluate the model on test-set
3. python export.py checkpoints/up_super_model-102-0.06.hdf5 # Export the model for deployment
4. python test.py test/four.jpeg # Test the model on a single image
5. python webcam.py test/beach.jpg # Run the model on webcam feed
6. python segvideo.py test/sunset.jpg # Apply blending filters on video
```
You may also run the **Jupyter Notebook** (ipynb) in google colaboratory, after downloading the training dataset.

In case you want to train with a **custom dataset**, check out the scripts in **utils** directory for data preparation.

## Training graphs

Since we are using a **pretrained mobilentv2** as encoder for a head start, the training **quickly converges to 90% accuracy** within first couple of epochs. Also, here we use a flexible **learning rate schedule** (ReduceLROnPlateau) for training the model.


### Training Loss

![Screenshot](graphs/train_loss.png)

### Training Accuracy

![Screenshot](graphs/train_acc.png)

### Validation Loss

![Screenshot](graphs/val_loss.png)

### Validation Accuracy

![Screenshot](graphs/val_acc.png)

### Learning Rate Schedule

![Screenshot](graphs/lr.png)


## Demo

### Result

Here the **inputs and outputs** are images of size **128x128**.
The **first row** represents the **input** and the **second row** shows the corresponding **cropped image** obtained by cropping the input image with the **mask output** of the model.

![Screenshot](result.png)

**Accuracy: 96%**

### Android Application

Real-time portrait video in android application

<p align="left">
  <img  src="android_portrait.gif">
</p>

(Shot on OnePlus 3 😉)

### Model running time (Android)

Summary of model running time and size 

|    | Model Name | CPU Time (ms) | GPU Time (ms)| Parameters (M) | Size (MB)
|----|----|----|----|-----|
| **deconv_fin_munet.tflite ** | 165 | 54  |  3.624 |  14.5 |
| **bilinear_fin_munet.tflite **  | 542  | 115 |  3.620 |  14.5 |

#### CPU Profiling :-

The benchmark tool allows us to profile the running time of **each operator in CPU** of the mobile device. Here is the summary of the operator profiling. 

**1. Deconv model (CPU)**

$ adb shell /data/local/tmp/benchmark_model_tf14 --graph=/data/local/tmp/deconv_fin_munet.tflite --enable_op_profiling=true --num_threads=1

```
Number of nodes executed: 94
============================== Summary by node type ==============================
	             [Node type]	  [count]	  [avg ms]	    [avg %]	    [cdf %]	  [mem KB]	[times called]
	          TRANSPOSE_CONV	        6	   130.565	    79.275%	    79.275%	     0.000	        6
	                     ADD	       21	    13.997	     8.499%	    87.773%	     0.000	       21
	                 CONV_2D	       34	     8.962	     5.441%	    93.215%	     0.000	       34
	                     MUL	        5	     7.022	     4.264%	    97.478%	     0.000	        5
	       DEPTHWISE_CONV_2D	       17	     3.177	     1.929%	    99.407%	     0.000	       17
	           CONCATENATION	        4	     0.635	     0.386%	    99.793%	     0.000	        4
	                     PAD	        5	     0.220	     0.134%	    99.927%	     0.000	        5
	                LOGISTIC	        1	     0.117	     0.071%	    99.998%	     0.000	        1
	                 RESHAPE	        1	     0.004	     0.002%	   100.000%	     0.000	        1

Timings (microseconds): count=50 first=164708 curr=162772 min=162419 max=167496 avg=164746 std=1434
```

**2. Bilinear model (CPU)**

$ adb shell /data/local/tmp/benchmark_model_tf14 --graph=/data/local/tmp/bilinear_fin_munet.tflite --enable_op_profiling=true --num_threads=1

```
Number of nodes executed: 84
============================== Summary by node type ==============================
	             [Node type]	  [count]	  [avg ms]	    [avg %]	    [cdf %]	  [mem KB]	[times called]
	                 CONV_2D	       39	   534.319	    98.411%	    98.411%	     0.000	       39
	         RESIZE_BILINEAR	        5	     3.351	     0.617%	    99.028%	     0.000	        5
	       DEPTHWISE_CONV_2D	       17	     3.110	     0.573%	    99.601%	     0.000	       17
	          TRANSPOSE_CONV	        1	     0.871	     0.160%	    99.761%	     0.000	        1
	           CONCATENATION	        4	     0.763	     0.141%	    99.901%	     0.000	        4
	                     PAD	        5	     0.246	     0.045%	    99.947%	     0.000	        5
	                     ADD	       11	     0.166	     0.031%	    99.977%	     0.000	       11
	                LOGISTIC	        1	     0.119	     0.022%	    99.999%	     0.000	        1
	                 RESHAPE	        1	     0.004	     0.001%	   100.000%	     0.000	        1

Timings (microseconds): count=50 first=544544 curr=540075 min=533873 max=551555 avg=542990 std=4363
```

* Unfortunately the benchmark tool doesn't allow gpu operator profiling. 

* For the current models, it  was observed that single threaded CPU execution was faster than multithreaded execution.

Note: All timings measured using [tflite benchmark tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark) on OnePlus3.


### Fun With Filters (Python)
<p align="justify">
  Let's add some filters to <b>harmonize</b> our output image with the background. Our aim is to give a <b>natural blended feel</b> to the output image i.e the <b>edges</b> should look smooth and the <b>lighting</b>(colour) of foreground should match(or blend) with its background.
 
 The first method is the traditional <b>alpha blending</b>, where the  foreground images are blended with background using the blurred(gaussian) version of the mask.In the <b>smooth-step</b> filter, we clamp the blurred edges and apply a  polynomial function to give a curved appearence to the foreground image edges.Next, we use the <b>colour transfer</b> algorithm to transfer the global colour to the foreground image.Also, opencv(computational photography) provides a function called <b>seamless clone</b> to blend an image onto a new background using an alpha mask.Finally, we use the dnn module of opencv to load a </b>colour harmonization</b> model(deep model) in <b>caffe</b> and transfer the background style to the foreground.
 </P>
 
 Here are some sample results:-

![Screenshot](blend_results.png)

For **live action**, checkout the script **segvideo.py** to see the effects applied on a **webcam video**.

Also download the **caffe model** and put it inside **models/caffe** folder.

#### Keyboard Controls:-

Hold **down** the following keys for **filter** selection.

* **C**- Colour transfer
* **S**- Seamless clone
* **M**- Smooth step
* **H**- Colour harmonize

Move the **slider** to change the background image.

### Tensorflowjs: Running the model on a browser
<p align="justify">
  To ensure that your applications runs in a <b>platform independent</b> way(portabe), the easiest way is to implement them as a <b>web-application</b> and run it using a <b>browser</b>.You can easily convert the trained model to tfjs format and run them using javascript with the help of tensorflowjs conversion tools.If you are familiar with <b>React/Vue</b> js , you can easily incorporate the tfjs into you application and come up with a really cool <b>AI webapp</b>, in no time!!!
</p>

Here is the **link** to the portrait segmentation **webapp**: [CVTRICKS](https://cvtricks.000webhostapp.com/)

If you want to run it **locally**, start a local server using python **SimpleHTTPServer**. Initially configure the **hostname, port and CORS permissions** and then run it using your browser. 

**NB:** The application is **computaionally intensive** and resource heavy.

## Key Insights and Drawbacks

1. Always start experimentation with **standard/pretrained** networks. Also try out **default/standard hyperparameter** settings before experimentation.
2. Make sure your **ground truth is correct/uncorrupted** and is in **desired format** before training (even standard dataset).
3. For **mobile devices**, make sure you use a **mobile-friendly architecture (like mobilenet)** for training and deployment.
4. Using **google colaboratory** along with google drive for training was **EASY & FUN**.It provides **high end GPU** (RAM also) for free.
5. Some of the mobile **optimization tools**(even TF) are still **experimental** (GPU deegate, NNAPI, FP16 etc.) and are buggy.They support only **limited operations and edge devices**.
6. Even **state-of-the art segmenation models**(deeplab-xception) seems to suffer from **false positives** (even at higher sizes), when we test them on a random image.
7. The **segmentaion maps** produced at this low resolution (128x128) have coarse or **sharp edges** (stair-case effect), especially when we resize them to higher resolution.
8. To tackle the problem of coarse edges, we apply a **blur filter** (also antialiasing at runtime) using **opencv** and perform **alpha blending** with the background image. Other approach was to **threshold** the blurred segmentation map with **smooth-step** function using **GLSL shaders**.
9. In android we can use **tensorflow-lite gpu-delegate** to speed up the inference.It was found that **flattening** the model output into a **rank 1 (or 2)** tensor helped us to reduce the **latency** due to **GPU-CPU** data transfer.Also this helped us to **post-process** the mask without looping over a multi-dimensional array.
10. Using **opencv (Android NEON)** for post-processing helped us to improve the **speed** of inference.But this comes at the cost of **additional memory** for opencv libraray in the application.
11. Still, there is a scope for improving the **latency** of inference by performing all the **postprocessing in the GPU**, without transfering the data to CPU. This can be acheived by using opengl shader storge buffers (**SSBO**). We can configure the GPU delegate to **accept input from SSBO** and also access model **output from GPU memory** for further processing (wihout CPU) and subsequent rendering.
12. Make sure most(all if possible) of your nodes or **layers** have a shape of the form **NHWC4** (i.e channels-C are multiple of four), if you plan to use **tflite gpu delegate**. This ensures that there are no **redundant memory copies** during shader execution. Similarly **avoid reshape** operators, whenever possible.These tricks can significanlty improve the overall **speed** or runnig time of your model on a mobile device(GPU).
13. The parameters like **image size, kernel size and strides** (ie. of form - [x,x]) have significant impact on model running time(especially cpu). Clearly, the **model layer running time** seems to be proportional to the **square of image and kernel size(x)** and **inversely  proportional to the square of stride values(x)**,(other params. being const. in each case). This could be mostly due to the proportional increase/decreae in **MAC operations**.
14. The **difference** between the **input image frame rate and output mask generation frame rate** may lead to an output(rendering), where the segmentation **mask lags behind current frame**.This **stale mask** phenemena arises due to the model(plus post-processing) taking more than 40ms (corr. to 25 fps input) per frame (real-time video). The **solution** is to render then output image in accordance to the **mask generation fps** (depends on device capability) or **reduce the input frame rate**.
15. If your segmentaion-mask output contains **minor artifacts**, you can clean them up using **morphological operations** like **opening or closing**. However it can be slightly **expensive** if your output image size is **large**, especially if you perform them on every frame output.
16. If the background consists of **noise, clutter or objects like clothes, bags**  etc. the model **fails** miserably.
17. Even though the stand-alone **running time** of exported (tflite) model is  **low(around 100 ms)**,other operations like **pre/post-processing, data loading, data-transfer** etc. consumes **significant time** in a mobile device.
18. The models trained with **resize bilinear**(default parameters) in tensorflow seems to suffer from a problem of **mask shifting**.This problem occurs if the image size is **even** (i.e bilinear_128 model in our case).The pixels in the output mask seems to be **sligtly shifted horizontaly** in one direction(left/right). 
19. **Opencv dnn** module provides support for running models trained on popular platforms like **Caffe,Tensorflow, Torch** etc.It supports acceleration through **OpenCL, Vulkan, Intel IE** etc.It also supports variety  of hardwares like **CPU,GPU and VPU**.Finally, we can also run  smaller **FP16 models** for improved speed.
20. Once you are familiar with tensorflow, it is fairly easy to **train and perform inference** using **tensorflowjs**.It also comes with support of **WebGL** backend for accelerating the **inference** and training process.Th main advantage is the **portability** of the application i.e it can be run on **PC, phones or tablet** without any modifications. 


## TODO

* Port the code to **TF 2.0**
* Use a **bigger image** for training(224x224)
* Try **quantization-aware** training
* Train with **mixed precision** (FP16) 
* Optimize the model by performing weight **pruning**
* Improve **accuracy** & reduce **artifacts** at runtime
* Incroporate **depth** information and **boundary refinement** techniques
* Apply **photorealistic style transfer** on foreground based on **background image**

## Versioning

Version 1.0

## Authors

Anil Sathyan

## Acknowledgments
* https://www.tensorflow.org/model_optimization
* https://github.com/cainxx/image-segmenter-ios
* https://github.com/gallifilo/final-year-project
* https://github.com/tantara/JejuNet
* https://github.com/lizhengwei1992/mobile_phone_human_matting
* https://github.com/dailystudio/ml/tree/master/deeplab
* https://github.com/yulu/GLtext
* https://machinethink.net/blog/mobilenet-v2/
* [On-Device Neural Net Inference with Mobile GPUs](https://arxiv.org/pdf/1907.01989.pdf)
*   [Deeplab Image Segmentation](https://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb)
*   [Tensorflow - Image segmentation](https://www.tensorflow.org/beta/tutorials/images/segmentation)
*   [Tensorflowjs - Tutorials](https://www.tensorflow.org/js)
*   [Hyperconnect - Tips for fast portrait segmentation](https://hyperconnect.github.io/2018/07/06/tips-for-building-fast-portrait-segmentation-network-with-tensorflow-lite.html)
* [Prismal Labs: Real-time Portrait Segmentation on Smartphones](https://blog.prismalabs.ai/real-time-portrait-segmentation-on-smartphones-39c84f1b9e66)
* [Keras Documentation](https://keras.io/)
* [Boundary-Aware Network for Fast and High-Accuracy Portrait Segmentation](https://arxiv.org/pdf/1901.03814.pdf)
* [Fast Deep Matting for Portrait Animation on Mobile Phone](https://arxiv.org/pdf/1707.08289.pdf)
* [Pyimagesearch - Super fast color transfer between images](https://www.pyimagesearch.com/2014/06/30/super-fast-color-transfer-images/)
* [Learn OpenCV - Seamless Cloning using OpenCV](https://www.learnopencv.com/seamless-cloning-using-opencv-python-cpp/)
* [Deep Image Harmonization](https://github.com/wasidennis/DeepHarmonization)
* [Tfjs Examples - Webcam Transfer Learning](https://github.com/tensorflow/tfjs-examples/blob/fc8646fa87de990a2fc0bab9d1268731186d9f04/webcam-transfer-learning/index.js)
* [Opencv Samples: DNN-Classification](https://github.com/opencv/opencv/blob/master/samples/dnn/classification.py)
* [Deep Learning In OpenCV](https://elinux.org/images/9/9e/Deep-Learning-in-OpenCV-Wu-Zhiwen-Intel.pdf)
* [BodyPix - Person Segmentation in the Browser](https://github.com/tensorflow/tfjs-models/tree/master/body-pix)
* [High-Resolution Network for Photorealistic Style Transfer](https://arxiv.org/pdf/1904.11617.pdf)
* [Tflite Benchmark Tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)
* [Ezgif: Online Image Editor](https://ezgif.com/)
* [Stackoverflow and Google](https://www.google.com/search?safe=active&sxsrf=ACYBGNTd70uFDhsbIL_sDXh5RlOpZtiWhQ%3A1570097540928&source=hp&ei=hMmVXaTvNcXerQH_j4DICA&q=stackoverflow&oq=stackoverflow&gs_l=psy-ab.3..35i39l2j0l2j0i20i263j0l3j0i131j0.1830.5666..5966...2.0..0.208.1381.12j1j1......0....1..gws-wiz.....10..35i362i39j0i67.tnbRuhKMNAk&ved=0ahUKEwikwb-R7f_kAhVFbysKHf8HAIkQ4dUDCAU&uact=5) 😜
