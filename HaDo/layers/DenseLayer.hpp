#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include <Eigen/Dense>
#include <vector>
#include "HaDo/base/Layer.hpp"
#include <memory>
#include <iostream>
#include <random>

using Eigen::Matrix;
using std::vector;
using std::cout;
using Eigen::Dynamic;
using std::unique_ptr;

namespace hado {

/**
 * @brief Dense layer class. T will only work for float, double,
 * or long double.
 * 
 * @tparam T Data type (float for speed, double accuracy) (optional)
*/
template<typename T=float> 
class DenseLayer : public Layer<T>{
private:

    // Convenience typedef
    using typename Layer<T>::MatrixD;

    // Convenience private variables
    int I{}, O{};

    // Weights and bias tensors
    MatrixD weights;
    MatrixD bias;

public:

    // Getters
    MatrixD getWeights() const { return weights; }
    MatrixD getBias() const { return bias; }

    /**
     * @brief Construct a new Dense Layer object
     * 
     * Initializes weights and bias with random values for this layer.
     * 
     * @param I rows/nodes in input tensor
     * @param O rows/nodes in output tensor
    */
    DenseLayer(const int I, const int O) : Layer<T>(1, 1, I, 1, O, 1){
        this->weights = MatrixD::Random(O, I);
        this->bias = MatrixD::Random(O, 1);
        this->inp = vector<MatrixD>(1);
        this->out = vector<MatrixD>(1);
        this->I = I;
        this->O = O;
    }

    // Copy constructor
    DenseLayer(const DenseLayer<T>& other) 
        : Layer<T>(1, 1, other.getInputRows(), 1, other.getOutputRows(), 1){
            this->weights = other.getWeights();
            this->bias = other.getBias();
            this->inp = other.inp;
            this->out = other.out;
            this->I = other.getInputRows();
            this->O = other.getOutputRows();
        }

    // Clone returning unique ptr
    std::unique_ptr<Layer<T>> clone() const override {
        return std::make_unique<DenseLayer<T>>(*this);
    }


    // Destructor
    ~DenseLayer() override = default;

    /**
     * @brief Forward pass of the dense layer. Input tensor must be a size 1 vector
     * of dimensions I x 1.
     * 
     * @param input_tensor Input tensor (one dimensional, must have right size)
     * @return vector<MatrixD> Output tensor
    */
    vector<MatrixD> forward(vector<MatrixD>& input_tensor) override {

        // Validity check
        this->assertInputDimensions(input_tensor);

        // Move input tensor into layer attribute inp
        this->inp = input_tensor;

        // Calculate output tensor
        auto res = (weights * (this->inp[0]) + bias);

        // Return output tensor
        return {res}; 
    }

    /**
     * @brief Backward pass of the dense layer. Output gradient tensor (input param) must be a size 1
     * vector<MatrixD> matching O rows.
     * 
     * @param output_gradient Output gradient tensor (one dimensional, must have right size)
     * @param learning_rate Learning rate for gradient descent (0 < learning_rate < 1)
     * @return vector<MatrixD> Input gradient tensor
    */
    vector<MatrixD> backward(vector<MatrixD>& output_gradient, const T learning_rate) override {

        // Validity check
        this->assertOutputDimensions(output_gradient);

        // Calculate weight gradient, bias gradient, and input gradient
        auto weight_gradient = (output_gradient[0]) * (this->inp[0]).transpose();
        auto bias_gradient = output_gradient[0];
        auto input_gradient = this->weights.transpose() * (output_gradient[0]);

        // Update weights and bias
        this->weights -= learning_rate * weight_gradient;
        this->bias -= learning_rate * bias_gradient;

        // Return input gradient
        return {input_gradient};
    }
};


}

#endif // DENSE_LAYER_HPP
