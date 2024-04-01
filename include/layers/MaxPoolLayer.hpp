#ifndef MAX_POOL_LAYER_HPP
#define MAX_POOL_LAYER_HPP

#include "Layer.hpp"
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <iostream>

using std::vector, Eigen::Matrix;


/**
 * @brief Max pooling layer class.
 * 
 * @details Max pooling layer class that performs max pooling on input tensor.
 * 
 * @tparam T Data type (float for speed, double accuracy) (optional)
*/
template <typename T=float>
class MaxPoolLayer : public Layer<T>{
private:
    int kernelSize;
    int stride;
    int padding; // True for padded (input size = output size), false unpadded - Data can be lost
    int prod;

    // Convenience typedef
    typedef Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixD;

public:

    // Getters
    int getKernelSize() const { return kernelSize; }
    int getStride() const { return stride; }
    int getPadding() const { return padding; }
    int getProd() const { return prod; }

    // Calculate output dimensions for convenience
    static constexpr int calcOutputRows(int inputRows, int kernelSize, int stride, int padding){
        return (inputRows - kernelSize + 2 * padding) / stride + 1;
    }

    static constexpr int calcOutputCols(int inputCols, int kernelSize, int stride, int padding){
        return (inputCols - kernelSize + 2 * padding) / stride + 1;
    }

    /**
     * @brief Construct a new Max Pool Layer object
     * 
     * @param inputDepth Depth of input tensor
     * @param inputRows Rows in input tensor
     * @param inputCols Columns in input tensor
     * @param kernelSize Size of kernel
     * @param stride Stride
     * @param padding Padding
    */
    MaxPoolLayer(int inputDepth, int inputRows, int inputCols, int kernelSize, int stride, int padding)
        : Layer<T>(inputDepth, inputDepth, inputRows, inputCols,
            (inputRows - kernelSize + 2 * padding) / stride + 1,
            (inputCols - kernelSize + 2 * padding) / stride + 1),
        kernelSize(kernelSize), stride(stride), padding(padding) {
            // Assert stride is positive
            if (stride <= 0){
                std::cerr << "Stride must be positive and non-zero." << endl;
                assert(stride > 0);
            }

            // Assert positive dimensions
            if (inputDepth <= 0 || inputRows <= 0 || inputCols <= 0){
                std::cerr << "Input dimensions must be positive and nonzero" << endl;
                assert(inputDepth > 0 && inputRows > 0 && inputCols > 0);
            }

            // Assert padding is non-negative
            if (padding < 0){
                std::cerr << "Padding must be non-negative." << endl;
                assert(padding >= 0);
            }

            // Assert kernel size is positive and not bigger than input size
            if (kernelSize <= 0 || kernelSize >= inputRows || kernelSize >= inputCols){
                std::cerr << "Kernel size must be positive and smaller than input size." << endl;
                assert(kernelSize > 0 && kernelSize < inputRows && kernelSize < inputCols);
            }

            prod = inputRows * inputCols;
        }

    // Copy constructor
    MaxPoolLayer(const MaxPoolLayer& other)
        : Layer<T>(other.getInputDepth(), other.getOutputDepth(), other.getInputRows(), other.getInputCols(),
            other.getOutputRows(), other.getOutputCols()),
        kernelSize(other.getKernelSize()), stride(other.getStride()), padding(other.getPadding()), prod(other.getProd()){}

    // Clone
    std::unique_ptr<Layer<T>> clone() const override {
        return std::make_unique<MaxPoolLayer>(*this);
    }

    // Destructor
    ~MaxPoolLayer() override {}

    // Forward pass
    vector<MatrixD> forward(vector<MatrixD> &input_tensor){

        // Assert input tensor dimensions
        this->assertInputDimensions(input_tensor);

        // Place input tensor in layer attribute
        this->inp = input_tensor;

        // Get depth
        const int depth = this->getInputDepth();

        // Initialize output tensor
        vector<MatrixD> output_tensor(depth);

        // Iterate over input tensor
        #ifdef _OPENMP
            #include <omp.h>
            if (depth > _MAX_DEPTH_UNTIL_THREADING && prod >= _MAX_PROD_UNTIL_THREADING){
                omp_set_num_threads(depth);
                #pragma omp parallel for
                for (int channel = 0; channel < depth; channel++){

                    // Initialize output matrix
                    MatrixD output(this->getOutputRows(), this->getOutputCols());

                    // Perform cross correlation
                    set_max_pool(input_tensor[channel], output, output_tensor[channel], this->out[channel], padding, stride, kernelSize);
                }
            } else{
                // Iterate over depth
                for (int channel = 0; channel < depth; channel++){

                    // Initialize output matrix
                    MatrixD output(this->getOutputRows(), this->getOutputCols());

                    // Perform cross correlation
                    set_max_pool(input_tensor[channel], output, output_tensor[channel], this->out[channel], padding, stride, kernelSize);
                }
            }
        #else
            // Iterate over depth
            for (int channel = 0; channel < depth; channel++){

                // Initialize output matrix
                MatrixD output(this->getOutputRows(), this->getOutputCols());

                // Perform cross correlation
                set_max_pool(input_tensor[channel], output, output_tensor[channel], this->out[channel], padding, stride, kernelSize);
            }
        #endif

        return output_tensor;
    }

    // Backward pass
    vector<MatrixD> backward(vector<MatrixD> &output_gradient, const T learning_rate) override {
        
        // Dimension check
        this->assertOutputDimensions(output_gradient);

        // Get depth
        const int depth = this->getOutputDepth();

        // Initialize input gradient tensor
        vector<MatrixD> input_gradient(depth);

        #ifdef _OPENMP
            #include <omp.h>
            if (depth > _MAX_DEPTH_UNTIL_THREADING && prod >= _MAX_PROD_UNTIL_THREADING){
                omp_set_num_threads(depth);
                #pragma omp parallel for
                for (int channel = 0; channel < depth; channel++){
                    backwards_max_pool(
                        this->inp[channel], 
                        output_gradient[channel], 
                        this->out[channel], 
                        input_gradient[channel], 
                        padding, stride, kernelSize);
                }
            } else{
                for (int channel = 0; channel < depth; channel++){
                    backwards_max_pool(
                        this->inp[channel], 
                        output_gradient[channel], 
                        this->out[channel], 
                        input_gradient[channel], 
                        padding, stride, kernelSize);
                }
            }
        #else
            for (int channel = 0; channel < depth; channel++){
                backwards_max_pool(
                    this->inp[channel], 
                    output_gradient[channel], 
                    this->out[channel], 
                    input_gradient[channel], 
                    padding, stride, kernelSize);
            }
        #endif

        return input_gradient;
    }

private:

    static constexpr auto set_max_pool(
        const MatrixD &input, 
        MatrixD &output, 
        MatrixD &output_location,
        MatrixD &output_copy_location,
        const int padding,
        const int stride,
        const int kernelSize){
            MatrixD padded_input;

            if (padding != 0){
                // Pad input tensor
                padded_input = MatrixD::Zero(input.rows() + 2 * padding, input.cols() + 2 * padding);

                // Copy input tensor to padded tensor
                padded_input.block(padding, padding, input.rows(), input.cols()) = input;
            } else{
                padded_input = input;
            }

            // Iterate over rows
            for (int i = 0; i < output.rows(); i++){
                // Iterate over columns
                for (int j = 0; j < output.cols(); j++){
                    // Get relevant input window
                    MatrixD input_window = padded_input.block(i * stride, j * stride, kernelSize, kernelSize);

                    // Find max value in window
                    T max_val = input_window.maxCoeff();

                    // Set max value in output matrix
                    output(i, j) = max_val;
                }
            }

            output_location = output;
            output_copy_location = output;
        }
    
    static constexpr auto backwards_max_pool(
        const MatrixD &original_input_channel,
        const MatrixD &output_gradient_channel,
        const MatrixD &output_channel,
        MatrixD &input_gradient_channel,
        const int padding,
        const int stride,
        const int kernelSize){

            MatrixD padded_input;
            MatrixD input_gradients = MatrixD::Zero(original_input_channel.rows(), original_input_channel.cols());

            if (padding != 0){
                // Pad input tensor
                padded_input = MatrixD::Zero(original_input_channel.rows() + 2 * padding, original_input_channel.cols() + 2 * padding);

                // Copy input tensor to padded tensor
                padded_input.block(padding, padding, original_input_channel.rows(), original_input_channel.cols()) = original_input_channel;
            } else{
                padded_input = original_input_channel;
            }

            // work backwards and match element of output channel to input channel and set gradient for that element
            // the gradient will be the same as the output gradient for the max value inside the window

            // Iterate over rows
            for (int i = 0; i < output_channel.rows(); i++){
                // Iterate over columns
                for (int j = 0; j < output_channel.cols(); j++){
                    // Get relevant input window
                    MatrixD input_window = padded_input.block(i * stride, j * stride, kernelSize, kernelSize);

                    // Find max value in window
                    int row, col;
                    input_window.maxCoeff(&row, &col);

                    if (i * stride + row < input_gradients.rows() && j * stride + col < input_gradients.cols()){
                        input_gradients(i * stride + row, j * stride + col) = output_gradient_channel(i, j);
                    }
                }
            }
            
            input_gradient_channel = input_gradients;
        }

};

#endif