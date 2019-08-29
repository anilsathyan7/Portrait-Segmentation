# Portrait-Segmentation


## Dependencies

* Tensorflow, Python 3
* Keras, Kito
* Opencv, PIL, Matplotlib

```
pip uninstall -y tensorflow
pip install -U tf-nightly
pip install keras
pip install kito
```

## Prerequisites

* Download training data-set
* GPU with CUDA support

## Demo

### Inputs



### Output

### TODO

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
