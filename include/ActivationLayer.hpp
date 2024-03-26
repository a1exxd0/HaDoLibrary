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
 * @tparam R Rows in input tensor
 * @tparam C Columns in input tensor
 * @tparam Activation Activation function
 * @tparam ActivationPrime Derivative of activation function
 * @tparam T Data type (float for speed, double accuracy) (optional)
*/
template<int D, int R, int C, typename Activation, typename ActivationPrime, typename T=float>
class ActivationLayer : public Layer<D, D, R, C, R, C, T> {
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
    ActivationLayer() : Layer<D, D, R, C, R, C, T>() {}

    // Destructor
    ~ActivationLayer() {}

    /**
     * @brief Forward pass of the activation layer. Input tensor must be a size 1 std::array
     * of std::unique_ptr<Matrix<T, D, 1>>. Output tensor is the same size.
     * 
     * @param input_tensor Input tensor (one dimensional, must have right size)
     * @return array<std::unique_ptr<Matrix<T, D, 1>>, 1> Output tensor
    */
    array<std::unique_ptr<Matrix<T, R, C>>, D> forward(
        array<std::unique_ptr<Matrix<T, R, C>>, D> input_tensor) {
            const int n = input_tensor.size();

            // Get copy because we need to pass one forward, and one stays in layer
            array<std::unique_ptr<Matrix<T, R, C>>, D> out_copy;

            // Iterate through depth of tensor
            for (int i = 0; i < n; i++){

                // Move input matrix into layer attribute inp
                this->inp[i] = std::move(input_tensor[i]);

                // Calculate output tensor for single layer
                auto res = std::make_unique<Matrix<T, R, C>>(this->inp[i]->unaryExpr(Activation()));

                // Copy output tensor to return
                out_copy[i] = std::make_unique<Matrix<T, R, C>>(*res);

                // Store a copy of the output tensor in layer attribute out
                this->out[i] = std::move(res);
            }

            return (std::move(out_copy));
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
    array<std::unique_ptr<Matrix<T, R, C>>, D> backward(
        array<std::unique_ptr<Matrix<T, R, C>>, D> output_gradient, T learning_rate) {
            const int n = output_gradient.size();

            // Array to store input gradient (not the input)
            array<std::unique_ptr<Matrix<T, R, C>>, D> input_gradient;

            // Iterate through depth of tensor
            for (int i = 0; i < n; i++){

                // Calculate input gradient for single layer
                auto res = std::make_unique<Matrix<T, R, C>>(
                    this->out[i]->unaryExpr(ActivationPrime())
                        .cwiseProduct(*(output_gradient[i])));

                // Store input gradient
                input_gradient[i] = std::move(res);
            }

            return {std::move(input_gradient)};
        }

};

#endif