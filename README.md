# A library for implementing Neural Networks in C++
Lightweight, templated, and optimised library for a variety of neural networks.

Supports models for:
 - Deep Neural Networks

Will support soon:
 - Convolutional Neural Networks
 - LSTM Networks

# Usage
You'll need the Eigen library inside of your project, as well as everything in the include/ subdirectory. See the Makefile for a general idea of how to compile it all.

# TODO
  - [ ] Image -> Bitmap formatting ??? Maybe use PPM for raw RGB
  - [x] Dense Layers
  - [x] Activation Layers (tanh, sigmoid, ReLu so far)
  - [X] Mean Squared Error implementation
  - [X] Method to pass through and verify layer setup
  - [ ] Pipeline class
  - [ ] Convolutional Layers
  - [ ] Pooling Layers
  - [ ] Saving a model (JSON or binary?)
  - [ ] Getting it running on a GPU
  - [ ] Getting it running on a cluster
