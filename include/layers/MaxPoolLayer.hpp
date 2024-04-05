#ifndef MAX_POOL_LAYER_HPP
#define MAX_POOL_LAYER_HPP

#include "Layer.hpp"
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <iostream>
#include <utility>

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
    int kernelSize; // Square kernel size
    int stride; // Stride size, number of elements to skip (min 1)
    int padding; // True for padded (input size = output size), false unpadded - Data can be lost
    int prod; // Convenience variable for input rows * input cols

    // Convenience typedef
    using typename Layer<T>::MatrixD;

public:

    // Getters
    int getKernelSize() const { return kernelSize; }
    int getStride() const { return stride; }
    int getPadding() const { return padding; }
    int getProd() const { return prod; }

    /**
     * @brief Calculate output rows and columns based on input tensor dimensions.
     * 
     * @param inputRows Rows in input tensor
     * @param inputCols Columns in input tensor
     * @param kernelSize Size of kernel
     * @param stride Stride
     * @param padding Padding
    */
    static constexpr std::pair<int, int> 
        calcOutputDimensions(int inputRows, int inputCols, int kernelSize, int stride, int padding){
            return std::make_pair(
                (inputRows - kernelSize + 2 * padding) / stride + 1,
                (inputCols - kernelSize + 2 * padding) / stride + 1);
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
    virtual std::unique_ptr<Layer<T>> clone() const override {
        return std::make_unique<MaxPoolLayer>(*this);
    }

    // Destructor
    ~MaxPoolLayer() override {}

    /**
     * @brief Forward pass of the max pooling layer.
     * 
     * @param input_tensor Input tensor
     * @return vector<MatrixD> Output tensor
    */
    virtual vector<MatrixD> forward(vector<MatrixD> &input_tensor) override {

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

                    // Perform max pool on single matrix
                    set_max_pool(input_tensor[channel], output, output_tensor[channel], this->out[channel], padding, stride, kernelSize);
                }
            } else{

                // Iterate over depth
                for (int channel = 0; channel < depth; channel++){

                    // Initialize output matrix
                    MatrixD output(this->getOutputRows(), this->getOutputCols());

                    // Perform max pool on single matrix
                    set_max_pool(input_tensor[channel], output, output_tensor[channel], this->out[channel], padding, stride, kernelSize);
                }
            }
        #else
            // Iterate over depth
            for (int channel = 0; channel < depth; channel++){

                // Initialize output matrix
                MatrixD output(this->getOutputRows(), this->getOutputCols());

                // Perform max pool on single matrix
                set_max_pool(input_tensor[channel], output, output_tensor[channel], this->out[channel], padding, stride, kernelSize);
            }
        #endif

        return output_tensor;
    }

    /**
     * @brief Backward pass of the max pooling layer.
     * 
     * @param output_gradient Output gradient tensor
     * @param learning_rate Learning rate (no effect here)
     * @return vector<MatrixD> Input gradient tensor
    */
    virtual vector<MatrixD> backward(vector<MatrixD> &output_gradient, const T learning_rate) override {
        
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

                    // For each channel, calculate input gradient matrix and set
                    backwards_max_pool(
                        this->inp[channel], 
                        output_gradient[channel], 
                        input_gradient[channel], 
                        padding, stride, kernelSize);
                }
            } else{
                for (int channel = 0; channel < depth; channel++){

                    // For each channel, calculate input gradient matrix and set
                    backwards_max_pool(
                        this->inp[channel], 
                        output_gradient[channel], 
                        input_gradient[channel], 
                        padding, stride, kernelSize);
                }
            }
        #else
            for (int channel = 0; channel < depth; channel++){

                // For each channel, calculate input gradient matrix and set
                backwards_max_pool(
                    this->inp[channel], 
                    output_gradient[channel], 
                    input_gradient[channel], 
                    padding, stride, kernelSize);
            }
        #endif

        return input_gradient;
    }

private:

    /**
     * @brief Set max pool for a single matrix.
     * 
     * @param input Input matrix from user
     * @param output Output matrix as a placeholder
     * @param output_location Where to store the output for return matrix
     * @param output_copy_location Output copy (in Layer::out)
     * @param padding Padding
     * @param stride Stride
     * @param kernelSize Kernel size
    */
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

            // Copy over results
            output_location = output;
            output_copy_location = output;
        }
    
    /**
     * @brief Backwards max pool for a single matrix.
     * 
     * @param original_input_channel Original input stored in forward pass
     * @param output_gradient_channel Output gradient matrix passed in for backpropagation
     * @param output_channel Output matrix stored in forward pass
     * @param input_gradient_channel Input gradient matrix to be set
     * @param padding Padding
     * @param stride Stride
     * @param kernelSize Kernel size
    */
    static constexpr auto backwards_max_pool(
        const MatrixD &original_input_channel,
        const MatrixD &output_gradient_channel,
        MatrixD &input_gradient_channel,
        const int padding,
        const int stride,
        const int kernelSize){

            // Pad the input and create temporary matrix for input gradients
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
            for (int i = 0; i < output_gradient_channel.rows(); i++){
                // Iterate over columns
                for (int j = 0; j < output_gradient_channel.cols(); j++){
                    // Get relevant input window from padded block
                    MatrixD input_window = padded_input.block(i * stride, j * stride, kernelSize, kernelSize);

                    // Find max value in window
                    int row, col;
                    input_window.maxCoeff(&row, &col);

                    // If the position it correlates to is valid (so we dont set padded portions/max values incorrectly)
                    // Set the gradient to the output gradient for that corresponding position
                    // Works in reverse of the forward pass
                    if (i * stride + row - padding < input_gradients.rows()
                        && i * stride + row - padding >= 0
                        && j * stride + col - padding < input_gradients.cols()
                        && j * stride + col - padding >= 0) {
                        input_gradients(i * stride + row - padding, j * stride + col - padding) = output_gradient_channel(i, j);
                    }
                }
            }
            
            input_gradient_channel = input_gradients;
        }

};

#endif