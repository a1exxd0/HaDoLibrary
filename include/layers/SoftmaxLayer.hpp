#ifndef SOFTMAX_LAYER_HPP
#define SOFTMAX_LAYER_HPP

#include <Eigen/Dense>
#include <vector>
#include "Layer.hpp"
#include <memory>
#include <thread>
#include <iostream>

using Eigen::Matrix;
using std::vector;
using std::cout;
using Eigen::Dynamic;
using std::unique_ptr;

/**
 * @brief Softmax layer class.
 * 
 * @tparam T Data type (float for speed, double accuracy) (optional)
*/
template<typename T=float>
class SoftmaxLayer : public Layer<T> {
private:

    // Convenience typedef
    typedef Matrix<T, Dynamic, Dynamic> MatrixD;

    // Number of rows in input tensor (and output)
    int rows;

    // Assert that T is either float, double, or long double at compiler time
    static_assert(
        std::is_same<T, float>::value 
        || std::is_same<T, double>::value
        || std::is_same<T, long double>::value,
        "T must be either float, double, or long double."
    );

public:

    /**
     * @brief Construct a new Softmax Layer object. Input and output dimensions
     * are the same, and input and output both have depth 1 and column count 1.
     * 
     * @param R Number of rows in input tensor
    */
    SoftmaxLayer(int R) : Layer<T>(1, 1, R, 1, R, 1) {
        rows = R;
    }

    // Copy constructor
    SoftmaxLayer(const SoftmaxLayer<T>& other) : Layer<T>(other) {
        rows = other.getInputRows();
    }

    // Clone
    unique_ptr<Layer<T>> clone() const {
        return std::make_unique<SoftmaxLayer<T>>(*this);
    }

    // Destructor
    ~SoftmaxLayer() {}

    /**
     * @brief Forward pass.
     * 
     * @param input_tensor Input tensor
     * @return vector<MatrixD> Output tensor
    */
    vector<MatrixD> forward(vector<MatrixD> &input_tensor){

        // Assert that input tensor has the correct dimensions
        this->assertInputDimensions(input_tensor);

        // Create output tensor location
        vector<MatrixD> output_tensor;

        // Calculate the exponential componentwise of the vector (tensor is vector here)
        MatrixD exp = input_tensor[0].array().exp();

        // Normalize the exponential vector
        output_tensor.push_back(exp / exp.sum());

        // Store output tensor
        this->out = output_tensor;

        return output_tensor;
    }

    /**
     * @brief Backward pass.
     * 
     * @param grad_tensor Gradient tensor of derivatives of loss w.r.t. output
     * @return vector<MatrixD> Gradient tensor of derivatives of loss w.r.t. input
    */
    vector<MatrixD> backward(vector<MatrixD> &grad_tensor, const T learning_rate){

        // Assert that gradient tensor has the correct dimensions
        this->assertOutputDimensions(grad_tensor);

        // Get the vector of outputs from softmax and replicate to form matrix
        MatrixD tiled = this->out[0].replicate(1, rows);

        // Calculate product of gradient vector and Jacobian of softmax output,
        // Which is the derivative of softmax w.r.t the input
        return {(tiled.cwiseProduct(MatrixD::Identity(rows, rows) - tiled.transpose())) * grad_tensor[0]};
    }
};

#endif