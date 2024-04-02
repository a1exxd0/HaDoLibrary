# A library for implementing Neural Networks in C++
Lightweight, templated, highly vectorised, and optimised library for a variety of neural networks.

Supports models for:
 - Deep Neural Networks

Will support soon:
 - Convolutional Neural Networks
 - RNN Networks
 - LSTM Networks
 - Transformer Networks

Example usages are stored in src folder to run. See "XorModel.cpp", for example. 

# Usage
You'll need the Eigen library inside of your project, as well as everything in the include/ subdirectory. See the Makefile for a general idea of how to compile it all. Use "-fopenmp" for compilation with multithreading enabled, if you have OpenMP installed on your system.

To use, all you need are the header files - import the neural network *.hpp file of your choice (i.e. "DeepNeuralNetwork.hpp") to get access to the relevant templates for use. Upcoming updates will enforce that you cant import modules that aren't top-level, i.e. individual layers - but can import types of networks, which provides you with the tools you need for that particular class of model.

Utilises optional multithreading via OpenMP if it is installed in the user system and compiled with -fopenmp flag.

# TODO
  - [X] Image -> Bitmap formatting ??? Maybe use PPM for raw RGB
  - [x] Dense Layers
  - [x] Activation Layers (tanh, sigmoid, ReLU, softmax so far)
  - [X] Mean Squared Error implementation
  - [X] Cross-Entropy Loss implementation
  - [X] Method to pass through and verify layer setup
  - [X] Pipeline class
  - [ ] Convolutional Layers
  - [X] Pooling Layers
  - [ ] Saving a model (JSON or binary?)
  - [ ] Getting it running on a GPU
  - [ ] Getting it running on a cluster
