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
 * @tparam T Data type (float for speed, double accuracy)
 * @tparam I Input vector length
 * @tparam O Output vector length
*/
template<typename T, int I, int O> 
class DenseLayer : public Layer<T, 1, 1, I, 1, O, 1>{
private:
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
    DenseLayer() : Layer<T, 1, 1, I, 1, O, 1>(){
        this->weights = Matrix<T, O, I>::Random();
        this->bias = Matrix<T, O, 1>::Random();
    }
    ~DenseLayer() {}

    array<std::unique_ptr<Matrix<T, O, 1>>, 1> forward(
        array<std::unique_ptr<Matrix<T, I, 1>>, 1> input_tensor) {
            this->inp = std::move(input_tensor);
            auto res = std::make_unique<Matrix<T, O, 1>>((weights * (*(this->inp[0])) + bias));

            array<std::unique_ptr<Matrix<T, O, 1>>, 1> out_copy;
            out_copy[0] = std::make_unique<Matrix<T, O, 1>>(*res);

            this->out[0] = std::move(res);
            return (out_copy); 
        }

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