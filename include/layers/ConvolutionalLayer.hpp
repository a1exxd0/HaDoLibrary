#ifndef CONVOLUTIONAL_LAYER_HPP
#define CONVOLUTIONAL_LAYER_HPP

#include "Layer.hpp"
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <iostream>


template<typename T>
class ConvolutionalLayer : public Layer<T> {
public:
    using MatrixD = typename Layer<T>::MatrixD;
    using Layer<T>::inp;
    using Layer<T>::out;

private:
    int kernelSize; // Size of the square kernel
    int stride; // Stride of the convolution
    int padding; // Padding applied to the input
    vector<MatrixD> filters; // Filters used for convolution

public:
    // Constructor
    ConvolutionalLayer(int inputDepth, int outputDepth, int inputRows, int inputCols,
                       int kernelSize, int stride, int padding)
    : Layer<T>(inputDepth, outputDepth, inputRows, inputCols,
               (inputRows - kernelSize + 2 * padding) / stride + 1, 
               (inputCols - kernelSize + 2 * padding) / stride + 1),
      kernelSize(kernelSize), stride(stride), padding(padding) {
        initializeFilters(outputDepth, inputDepth, kernelSize);
    }

    // Initialize filters with random values
    void initializeFilters(int numFilters, int depth, int size) {
        filters.resize(numFilters);
        for(auto& filter : filters) {
            filter = MatrixD::Random(size, size);
        }
    }

    virtual vector<MatrixD> forward(vector<MatrixD>& input_tensor) override {
        // Placeholder
    }

    virtual vector<MatrixD> backward(vector<MatrixD>& output_gradient, T learning_rate) override {
        // Placeholder
    }
};

#endif // CONVOLUTIONAL_LAYER_HPP