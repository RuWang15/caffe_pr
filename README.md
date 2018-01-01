
# Caffe for Pseudo-Random Dropout
Our modified Caffe for pseudo-random dropout. This repository at least runs on Ubuntu 14.04, gcc 4.8, OpenCV 2.4.10, CUDA 8.0, and CUDNN 5.
## New Features
- Add [inner product dropout layer](https://github.com/WilliamRuRu15/caffe_pr/blob/master/src/caffe/layers/inner_product_dropout_layer.cu) for GPU acceleration.
  - Up to 2.4 times as fast as dropout + inner product. (forward pass, test on GTX 1080)
- New functions in [math_functions.cu](https://github.com/WilliamRuRu15/caffe_pr/blob/master/src/caffe/util/math_functions.cu) to optimize matrix multiplication.
## Illustration
- The overall process from sequence generating to forward pass of IPD layer is shown as below. Note that this repository does **not** contain the part in dashed line box.
For code or files of that part, please see [Pseudo-Random-Dropout](https://github.com/rudongyu/Pseudo-Random-Dropout) for detail.
![illustration](https://github.com/WilliamRuRu15/caffe_pr/blob/master/images/pseudo-random_illustration.png)
## Installation
- You may add `CXXFLAGS += -std=c++11` in Makefile.Config before compiling.
- For detailed installation instructions of Caffe please search on the internet. For convenience (not recommended) you just need one command `make all -j8`.
## Example 
- The example of usage of inner product dropout layer is as follows:
    - `seq_addr` is the address of the skipping period sequence file. The file is formatted as in [seq_in1024.txt](https://github.com/WilliamRuRu15/caffe_pr/blob/master/seq_in1024.txt). The first number denotes the length of the sequence, and the rest numbers are elements.
    - Other parameters are the same with that of inner product layer.
```
layer {
  name: "ipd1"
  type: "InnerProductDropout"
  bottom: "pool3"
  top: "ipd1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_dropout_param {
    seq_addr: "/home/user/caffe_pr/seq_in1024.txt"
    num_output: 64
    weight_filler {
      type: "gaussian"
      std: 0.1
    }
    bias_filler {
      type: "constant"
    }
  }
}
```
### More examples and test files will be updated soon!
# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by Berkeley AI Research ([BAIR](http://bair.berkeley.edu))/The Berkeley Vision and Learning Center (BVLC) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BAIR reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

## Custom distributions

 - [Intel Caffe](https://github.com/BVLC/caffe/tree/intel) (Optimized for CPU and support for multi-node), in particular Xeon processors (HSW, BDW, SKX, Xeon Phi).
- [OpenCL Caffe](https://github.com/BVLC/caffe/tree/opencl) e.g. for AMD or Intel devices.
- [Windows Caffe](https://github.com/BVLC/caffe/tree/windows)

## Community

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BAIR/BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
