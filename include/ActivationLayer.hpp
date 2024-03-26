#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

#include <Eigen/Dense>
#include <array>
#include "Layer.hpp"
#include <memory>
#include <iostream>

using Eigen::Matrix;
using std::array;

/**
 * @brief Activation layer class.
 * 
 * @details Activation layer class that applies an activation function
 * 
 * @tparam D Depth of input tensor
 * @tparam Activation Activation function
 * @tparam ActivationPrime Derivative of activation function
 * @tparam T Data type (float for speed, double accuracy)
*/
template<int D, typename Activation, typename ActivationPrime, typename T=float>
class ActivationLayer : public Layer<1, 1, D, 1, D, 1, T> {
private:

    // Assert that T is either float, double, or long double at compiler time
    static_assert(
        std::is_same<T, float>::value 
        || std::is_same<T, double>::value
        || std::is_same<T, long double>::value,
        "T must be either float, double, or long double."
    );

    // Assert that Activation and ActivationPrime are functions that take a scalar and return a scalar
    static_assert(
        std::is_invocable_r<T, Activation, T>::value,
        "Activation must be a function that takes a scalar and returns a scalar."
    );
    static_assert(
        std::is_invocable_r<T, ActivationPrime, T>::value,
        "ActivationPrime must be a function that takes a scalar and returns a scalar."
    );

public:

    /**
     * @brief Construct a new Activation Layer object
    */
    ActivationLayer() : Layer<1, 1, D, 1, D, 1, T>() {}

    // Destructor
    ~ActivationLayer() {}

    /**
     * @brief Forward pass of the activation layer. Input tensor must be a size 1 std::array
     * of std::unique_ptr<Matrix<T, D, 1>>. Output tensor is the same size.
     * 
     * @param input_tensor Input tensor (one dimensional, must have right size)
     * @return array<std::unique_ptr<Matrix<T, D, 1>>, 1> Output tensor
    */
    array<std::unique_ptr<Matrix<T, D, 1>>, 1> forward(
        array<std::unique_ptr<Matrix<T, D, 1>>, 1> input_tensor) {
            this->inp = std::move(input_tensor);
            auto res = std::make_unique<Matrix<T, D, 1>>(this->inp[0]->unaryExpr(Activation()));

            array<std::unique_ptr<Matrix<T, D, 1>>, 1> out_copy;
            out_copy[0] = std::make_unique<Matrix<T, D, 1>>(*res);

            this->out[0] = std::move(res);
            return (out_copy);
        }

    /**
     * @brief Backward pass of the activation layer. Output gradient tensor must be a size 1 std::array
     * of std::unique_ptr<Matrix<T, D, 1>>. Input gradient tensor (returned) is the same size.
     * 
     * @param output_gradient Output gradient tensor (one dimensional, must have right size)
     * @param learning_rate Learning rate
     * @return array<std::unique_ptr<Matrix<T, D, 1>>, 1> Input gradient tensor
     */
    #pragma GCC diagnostic ignored "-Wunused-parameter"
    array<std::unique_ptr<Matrix<T, D, 1>>, 1> backward(
        array<std::unique_ptr<Matrix<T, D, 1>>, 1> output_gradient, T learning_rate) {
            auto input_gradient = std::make_unique<Matrix<T, D, 1>>(
                this->out[0]->unaryExpr(ActivationPrime())
                    .cwiseProduct(*(output_gradient[0])));

            return {std::move(input_gradient)};
        }

};

#endif