#ifndef SOFTMAX_LAYER_HPP
#define SOFTMAX_LAYER_HPP

#include <Eigen/Dense>
#include <vector>
#include "HaDo/base/Layer.hpp"
#include <memory>
#include <iostream>

using Eigen::Matrix;
using std::vector;
using std::cout;
using Eigen::Dynamic;
using std::unique_ptr;

namespace hado {

/**
 * @brief Softmax layer class.
 * 
 * @tparam T Data type (float for speed, double accuracy) (optional)
*/
template<typename T=float>
class SoftmaxLayer : public Layer<T> {
private:

    // Convenience typedef
    using typename Layer<T>::MatrixD;

    // Number of rows in input tensor (and output)
    int rows;

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
    virtual unique_ptr<Layer<T>> clone() const override {
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
    virtual vector<MatrixD> forward(vector<MatrixD> &input_tensor) override {

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
    virtual vector<MatrixD> backward(vector<MatrixD> &grad_tensor, const T learning_rate) override {

        // Assert that gradient tensor has the correct dimensions
        this->assertOutputDimensions(grad_tensor);

        // Get the vector of outputs from softmax and replicate to form matrix
        MatrixD tiled = this->out[0].replicate(1, rows);

        // Calculate product of gradient vector and Jacobian of softmax output,
        // Which is the derivative of softmax w.r.t the input
        return {(tiled.cwiseProduct(MatrixD::Identity(rows, rows) - tiled.transpose())) * grad_tensor[0]};
    }
};

}


#endif
