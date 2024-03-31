#ifndef CONVOLUTIONAL_LAYER_HPP
#define CONVOLUTIONAL_LAYER_HPP

#include "Layer.hpp"
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <iostream>

using std::vector;

template <typename T, typename ActivationFunc, typename ActivationFuncPrime>
class ConvolutionalLayer : public Layer<T>
{
private:
    int kernelSize; // Size of the square kernel
    int stride;     // Stride of the convolution
    int padding;    // Padding applied to the input
    int inputDepth, outputDepth, inputRows, inputCols, outputRows, outputCols;
    typedef Matrix<T, Dynamic, Dynamic> MatrixD; // Convenience typedef
    vector<vector<MatrixD>> filters;             // Filters used for the convolution

    // Assert that Activation and ActivationPrime are functions that take a scalar and return a scalar
    static_assert(
        std::is_invocable_r<T, Activation, T>::value,
        "Activation must be a function that takes a scalar and returns a scalar."
    );
    static_assert(
        std::is_invocable_r<T, ActivationPrime, T>::value,
        "ActivationPrime must be a function that takes a scalar and returns a scalar."
    );

    // Assert that kernelSize, stride, and padding are positive and nonzero
    // Assertions for dimensions are in layer class
    static_assert(kernelSize > 0, "kernelSize must be positive and nonzero.");
    static_assert(stride > 0, "stride must be positive and nonzero.");
    static_assert(padding >= 0, "padding must be positive.");


public:
    using Layer<T>::inp;
    using Layer<T>::out;

    // Getters
    int getKernelSize() const { return kernelSize; }
    int getStride() const { return stride; }
    int getPadding() const { return padding; }
    vector<vector<MatrixD>> getFilters() const { return filters; }

    // Constructor
    ConvolutionalLayer(int inputDepth, int outputDepth, int inputRows, int inputCols,
                       int kernelSize, int stride, int padding)
        : Layer<T>(inputDepth, outputDepth, inputRows, inputCols,
                   (inputRows - kernelSize + 2 * padding) / stride + 1,
                   (inputCols - kernelSize + 2 * padding) / stride + 1),
          kernelSize(kernelSize), stride(stride), padding(padding),
          inputDepth(inputDepth), outputDepth(outputDepth),
          inputRows(inputRows), inputCols(inputCols)
    {
        // Initialize output dimensions based on constructor parameters
        this->outputRows = (inputRows - kernelSize + 2 * padding) / stride + 1;
        this->outputCols = (inputCols - kernelSize + 2 * padding) / stride + 1;

        initializeFilters(outputDepth, inputDepth, kernelSize);
    }

    // Copy constructor
    ConvolutionalLayer(const ConvolutionalLayer &cl)
        : Layer<T>(cl.getInputDepth(), cl.getOutputDepth(), cl.getInputRows(), cl.getInputCols(),
        cl.getOutputRows(), cl.getOutputCols()){
            this->kernelSize = cl.getKernelSize();
            this->stride = cl.getStride();
            this->padding = cl.getPadding();
            this->filters = cl.getFilters();
        }

    // Clone returning unique ptr
    std::unique_ptr<Layer<T>> clone() const override
    {
        return std::make_unique<ConvolutionalLayer>(*this);
    }

    // Destructor
    ~ConvolutionalLayer() override {}

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
        // Iterate over filters, apply (convolve) each filter matrix within a filter to its respective matrix in the input
        // Sum up convolved matrices to get output matrix for the current vector of filters
        // Apply activation function to each output matrix
        // Output vector will be same length as number of filters
        this->inp = input_tensor;         // Store the input tensor for potential backward passes or inspection
        this->out.clear();                // Clear previous outputs
        this->out.resize(filters.size()); // Resize output vector to hold a feature map for each filter
        // Iterate over each filter
        for (size_t filterIndex = 0; filterIndex < filters.size(); ++filterIndex)
        {
            auto &filter = filters[filterIndex];
            MatrixD outputFeatureMap = MatrixD::Zero(this->getOutputRows(), this->getOutputCols()); // Initialize output feature map for this filter

            // Each filter has a matrix for each channel in the input tensor
            for (size_t channel = 0; channel < filter.size(); ++channel)
            {
                // Perform convolution between the channel of the input tensor and the corresponding matrix in the filter
                // The result is added to the output feature map
                outputFeatureMap += convolve(input_tensor[channel], filter[channel]);
            }

            // Apply activation function to the output feature map
            outputFeatureMap = outputFeatureMap.unaryExpr(ActivationFunc());

            // Store the activated feature map
            this->out[filterIndex] = outputFeatureMap;
        }

        return this->out; // Return the vector of output feature maps
    }

    virtual vector<MatrixD> backward(vector<MatrixD> &output_gradient, T learning_rate) override
    {
        // Placeholder
        return vector<MatrixD>();
    }

private:
    #pragma GCC push_options
    #pragma GCC optimize("O3")
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
    #pragma GCC pop_options

    // REWRITING CONVOLVE to utilisemore of the libr/ SIMD
    MatrixD convolve(const MatrixD& input, const MatrixD& kernel){

    }
};

#endif // CONVOLUTIONAL_LAYER_HPP