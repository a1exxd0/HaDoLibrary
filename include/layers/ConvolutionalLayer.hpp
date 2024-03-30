#ifndef CONVOLUTIONAL_LAYER_HPP
#define CONVOLUTIONAL_LAYER_HPP

#include "Layer.hpp"
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <iostream>

using std::vector;

template <typename T>
class ConvolutionalLayer : public Layer<T>
{
public:
    using MatrixD = typename Layer<T>::MatrixD;
    using Layer<T>::inp;
    using Layer<T>::out;

private:
    int kernelSize;                  // Size of the square kernel
    int stride;                      // Stride of the convolution
    int padding;                     // Padding applied to the input
    vector<vector<MatrixD>> filters; // Filters used for the convolution

public:
    // Constructor
    ConvolutionalLayer(int inputDepth, int outputDepth, int inputRows, int inputCols,
                       int kernelSize, int stride, int padding)
        : Layer<T>(inputDepth, outputDepth, inputRows, inputCols,
                   (inputRows - kernelSize + 2 * padding) / stride + 1,
                   (inputCols - kernelSize + 2 * padding) / stride + 1),
          kernelSize(kernelSize), stride(stride), padding(padding)
    {
        initializeFilters(outputDepth, inputDepth, kernelSize);
    }

    // Copy constructor
    ConvolutionalLayer(const ConvolutionalLayer &convLayer)
        : Layer<T>(convLayer.getDepth(), convLayer.getDepth(), convLayer.getRows(), convLayer.getCols(),
                   convLayer.getOutRows(), convLayer.getOutCols()),
          kernelSize(convLayer.kernelSize), stride(convLayer.stride), padding(convLayer.padding),
          filters(convLayer.filters) {}

    // Clone returning unique ptr
    std::unique_ptr<Layer<T>> clone() const override
    {
        return std::make_unique<ConvolutionalLayer>(*this);
    }

    // Initialize filters with random values
    void initializeFilters(int numFilters, int depth, int size)
    {
        filters.resize(numFilters);
        for (auto &filter : filters)
        {
            filter.resize(depth); // Each filter has 'depth' matrices
            for (auto &matrix : filter)
            {
                matrix = MatrixD::Random(size, size); // Initialize each matrix
            }
        }
    }

    virtual vector<MatrixD> forward(vector<MatrixD> &input_tensor) override
    {
        // Plan:
        // Iterate over filters, apply (convolve) each filter matrix within a filter to it's respective matrix in the input
        // Sum up convolved matrices to get output matrix for the current vector of filters
        // Output vector will be same length as number of filters
    }

    virtual vector<MatrixD> backward(vector<MatrixD> &output_gradient, T learning_rate) override
    {
        // Placeholder
    }

private:
    MatrixD convolve(const MatrixD &input, const MatrixD &kernel)
    {
        // Calculate modified input dimensions
        int modifiedRows = input.rows() + 2 * padding;
        int modifiedCols = input.cols() + 2 * padding;

        // Initialize modified input with zeros
        MatrixD modifiedInput = MatrixD::Zero(modifiedRows, modifiedCols);
        // Copy input into the center of modifiedInput
        modifiedInput.block(padding, padding, input.rows(), input.cols()) = input;

        // Determine output dimensions
        int outputRows = 1 + (modifiedRows - kernel.rows()) / stride;
        int outputCols = 1 + (modifiedCols - kernel.cols()) / stride;
        MatrixD output = MatrixD::Zero(outputRows, outputCols);

        // Perform convolution
        for (int y = 0; y < outputRows; ++y)
        {
            for (int x = 0; x < outputCols; ++x)
            {
                // For each position in the output, calculate the convolution
                // over the kernel size
                for (int ky = 0; ky < kernel.rows(); ++ky)
                {
                    for (int kx = 0; kx < kernel.cols(); ++kx)
                    {
                        // Calculate the indices on the modified input
                        int iy = y * stride + ky;
                        int ix = x * stride + kx;
                        output(y, x) += modifiedInput(iy, ix) * kernel(ky, kx);
                    }
                }
            }
        }

        return output;
    }
};

#endif // CONVOLUTIONAL_LAYER_HPP