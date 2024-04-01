#ifndef MAX_POOL_LAYER_HPP
#define MAX_POOL_LAYER_HPP

#include "Layer.hpp"
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <iostream>

using std::vector, Eigen::Matrix;

template <typename T>
class MaxPoolLayer : public Layer<T>{
private:
    int kernelSize;
    int stride;
    int padding; // True for padded (input size = output size), false unpadded - Data can be lost

    // Convenience typedef
    typedef Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixD;

public:

    // Getters
    int getKernelSize() const { return kernelSize; }
    int getStride() const { return stride; }
    int getPadding() const { return padding; }

    // Calculate output dimensions for convenience
    static constexpr int calcOutputRows(int inputRows, int kernelSize, int stride, int padding){
        return (inputRows - kernelSize + 2 * padding) / stride + 1;
    }

    static constexpr int calcOutputCols(int inputCols, int kernelSize, int stride, int padding){
        return (inputCols - kernelSize + 2 * padding) / stride + 1;
    }

    // Constructor
    MaxPoolLayer(int inputDepth, int inputRows, int inputCols, int kernelSize, int stride, int padding)
        : Layer<T>(inputDepth, inputDepth, inputRows, inputCols,
            (inputRows - kernelSize + 2 * padding) / stride + 1,
            (inputCols - kernelSize + 2 * padding) / stride + 1),
        kernelSize(kernelSize), stride(stride), padding(padding) {
            // Assert stride is positive
            if (stride <= 0){
                std::cerr << "Stride must be positive and non-zero." << std::endl;
                assert(stride > 0);
            }

            // Assert positive dimensions
            if (inputDepth <= 0 || inputRows <= 0 || inputCols <= 0){
                std::cerr << "Input dimensions must be positive and nonzero" << std::endl;
                assert(inputDepth > 0 && inputRows > 0 && inputCols > 0);
            }

            // Assert padding is non-negative
            if (padding < 0){
                std::cerr << "Padding must be non-negative." << std::endl;
                assert(padding >= 0);
            }

            // Assert kernel size is positive and not bigger than input size
            if (kernelSize <= 0 || kernelSize >= inputRows || kernelSize >= inputCols){
                std::cerr << "Kernel size must be positive and smaller than input size." << std::endl;
                assert(kernelSize > 0 && kernelSize < inputRows && kernelSize < inputCols);
            }
        }

    // Copy constructor
    MaxPoolLayer(const MaxPoolLayer& other)
        : Layer<T>(other.getInputDepth(), other.getOutputDepth(), other.getInputRows(), other.getInputCols(),
            other.getOutputRows(), other.getOutputCols()),
        kernelSize(other.kernelSize), stride(other.stride), padding(other.padding) {}

    // Clone
    std::unique_ptr<Layer<T>> clone() const override {
        return std::make_unique<MaxPoolLayer>(*this);
    }

    // Destructor
    ~MaxPoolLayer() override {}

    // Forward pass
    vector<MatrixD> forward(vector<MatrixD> &input_tensor){

        // Assert input tensor dimensions
        if (input_tensor.size() != (size_t) this->getInputDepth() 
            || input_tensor[0].rows() != this->getInputRows() 
            || input_tensor[0].cols() != this->getInputCols()){
            std::cerr << "Expected depth " << this->I << " but got depth " << input_tensor.size() << std::endl;
            std::cerr << "Expected rows " << this->RI << " but got rows " << input_tensor[0].rows() << std::endl;
            std::cerr << "Expected cols " << this->CI << " but got cols " << input_tensor[0].cols() << std::endl;
            throw std::invalid_argument("Input tensor match dimensions of layer.");
        }

        // Get depth
        const int depth = this->getInputDepth();

        // Initialize output tensor
        vector<MatrixD> output_tensor(depth);

        // Iterate over input tensor
        for (int channel = 0; channel < depth; channel++){
            // Get relevant input channel
            const MatrixD& input_channel = input_tensor[channel];

            MatrixD padded_input;
            if (padding != 0){
                // Pad input tensor
                padded_input = MatrixD::Zero(this->getInputRows() + 2 * padding, this->getInputCols() + 2 * padding);

                // Copy input tensor to padded tensor
                padded_input.block(padding, padding, this->getInputRows(), this->getInputCols()) = input_channel;
            } else{
                padded_input = input_channel;
            }


            // Initialize output matrix for this channel
            MatrixD output(this->getOutputRows(), this->getOutputCols());
            


        }

        return output_tensor;
    }


};

#endif