#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include <Eigen/Dense>
#include <array>
#include "Layer.hpp"
#include <memory>
#include <iostream>

using Eigen::Matrix;
using std::array;
using std::cout;

/**
 * @brief Dense layer class. T will only work for float, double,
 * or long double.
 * 
 * @tparam I Input vector length
 * @tparam O Output vector length
 * @tparam T Data type (float for speed, double accuracy)
*/
template<int I, int O, typename T=float> 
class DenseLayer : public Layer<1, 1, I, 1, O, 1, T>{
private:

    // Check if T is float, double, or long double
    static_assert(
        std::is_same<T, float>::value 
        || std::is_same<T, double>::value
        || std::is_same<T, long double>::value,
        "T (first template param) must be either float, double, or long double."
    );

    // Weights and bias tensors
    Matrix<T, O, I> weights;
    Matrix<T, O, 1> bias;

public:

    /**
     * @brief Construct a new Dense Layer object
     * 
     * Initializes weights and bias with random values for this layer.
    */
    DenseLayer() : Layer<1, 1, I, 1, O, 1, T>(){
        this->weights = Matrix<T, O, I>::Random();
        this->bias = Matrix<T, O, 1>::Random();
    }

    // Destructor
    ~DenseLayer() {}

    /**
     * @brief Forward pass of the dense layer. Input tensor must be a size 1 std::array
     * of std::unique_ptr<Matrix<T, I, 1>>.
     * 
     * @param input_tensor Input tensor (one dimensional, must have right size)
     * @return array<std::unique_ptr<Matrix<T, O, 1>>, 1> Output tensor
    */
    array<std::unique_ptr<Matrix<T, O, 1>>, 1> forward(
        array<std::unique_ptr<Matrix<T, I, 1>>, 1> input_tensor) {
            this->inp = std::move(input_tensor);
            auto res = std::make_unique<Matrix<T, O, 1>>((weights * (*(this->inp[0])) + bias));

            array<std::unique_ptr<Matrix<T, O, 1>>, 1> out_copy;
            out_copy[0] = std::make_unique<Matrix<T, O, 1>>(*res);

            this->out[0] = std::move(res);
            return (out_copy); 
        }

    /**
     * @brief Backward pass of the dense layer. Output gradient tensor must be a size 1 std::array
     * of std::unique_ptr<Matrix<T, O, 1>>.
     * 
     * @param output_gradient Output gradient tensor (one dimensional, must have right size)
     * @param learning_rate Learning rate for gradient descent (0 < learning_rate < 1)
     * @return array<std::unique_ptr<Matrix<T, I, 1>>, 1> Input gradient tensor
    */
    array<unique_ptr<Matrix<T, I, 1>>, 1> backward(
        array<unique_ptr<Matrix<T, O, 1>>, 1> output_gradient, T learning_rate){
            auto weight_gradient = (*(output_gradient[0])) * (*(this->inp[0])).transpose();
            auto bias_gradient = *(output_gradient[0]);
            auto input_gradient = std::make_unique<Matrix<T, I, 1>>(
                this->weights.transpose() * (*(output_gradient[0])));

            this->weights -= learning_rate * weight_gradient;
            this->bias -= learning_rate * bias_gradient;

            return {std::move(input_gradient)};
        }
};

#endif // DENSE_LAYER_HPP