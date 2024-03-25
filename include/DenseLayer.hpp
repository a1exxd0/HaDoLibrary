#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include <Eigen/Dense>
#include <array>
#include "Layer.hpp"
#include <memory>

using Eigen::Matrix;
using std::array;


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
    DenseLayer() : Layer<T, 1, 1, I, 1, O, 1>(){}
    ~DenseLayer() {}

    array<std::unique_ptr<Matrix<T, O, 1>>, 1> forward(array<std::unique_ptr<Matrix<T, I, 1>>, 1> input_tensor) {
        this->inp = std::move(input_tensor);
        auto res = std::make_unique<Matrix<T, O, 1>>((weights * (*input_tensor[0]) + bias));
        this->out[0] = std::move(res);
        return std::move(this->out); 
    }

    T backward(T output_gradient, T learning_rate) {
        return learning_rate;
    }
};

#endif // DENSE_LAYER_HPP