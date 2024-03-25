#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

#include <Eigen/Dense>
#include <array>
#include "Layer.hpp"
#include <memory>
#include <iostream>

using Eigen::Matrix;
using std::array;


template<typename T, int D, typename Activation, typename ActivationPrime>
class ActivationLayer : public Layer<T, 1, 1, D, 1, D, 1> {
private:
    static_assert(
        std::is_same<T, float>::value 
        || std::is_same<T, double>::value
        || std::is_same<T, long double>::value,
        "T must be either float, double, or long double."
    );

    static_assert(
        std::is_invocable_r<T, Activation, T>::value,
        "Activation must be a function that takes a scalar and returns a scalar."
    );

    static_assert(
        std::is_invocable_r<T, ActivationPrime, T>::value,
        "ActivationPrime must be a function that takes a scalar and returns a scalar."
    );

public:
    ActivationLayer() : Layer<T, 1, 1, D, 1, D, 1>() {}
    ~ActivationLayer() {}

    array<std::unique_ptr<Matrix<T, D, 1>>, 1> forward(
        array<std::unique_ptr<Matrix<T, D, 1>>, 1> input_tensor) {
            this->inp = std::move(input_tensor);
            auto res = std::make_unique<Matrix<T, D, 1>>(this->inp[0]->unaryExpr(Activation()));

            array<std::unique_ptr<Matrix<T, D, 1>>, 1> out_copy;
            out_copy[0] = std::make_unique<Matrix<T, D, 1>>(*res);

            this->out[0] = std::move(res);
            return (out_copy);
        }

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