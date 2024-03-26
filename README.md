# A library for implementing Convolutional Neural Networks
Lightweight, templated, and optimised library for custom CNN for image classification.

# Usage
You'll need the Eigen library inside of your project, as well as everything in the include/ subdirectory. See the Makefile for a general idea of how to compile it all.

# TODO
  - [ ] Image -> Bitmap formatting ??? Maybe use PPM for raw RGB
  - [x] Dense Layers
  - [x] Activation Layers (tanh, sigmoid, ReLu so far)
  - [X] Mean Squared Error implemented
  - [ ] Convolutional Layers
  - [ ] Pooling Layers
  - [ ] Saving a model (JSON or binary?)
  - [ ] Getting it running on a GPU
  - [ ] Getting it running on a cluster