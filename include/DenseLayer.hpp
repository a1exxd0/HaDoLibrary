#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include <Eigen/Dense>
#include <Layer.hpp>

/**
 * @brief Dense layer class. T will only work for float, double,
 * or long double.
 * 
 * @tparam T Data type (float for speed, double accuracy)
 * @tparam I Input vector length
 * @tparam O Output vector length
*/
template<typename T, int I, int O> 
class DenseLayer : public Layer<float, 1, 1> {
private:
    static_assert(
        std::is_same<T, float>::value 
        || std::is_same<T, double>::value
        || std::is_same<T, long double>::value,
        "T (first template param) must be either float, double, or long double."
    );

    // Input and output tensors
    typename Eigen::Tensor<T, I> inp;
    typename Eigen::Tensor<T, O> out;

    // Weights and bias tensors
    typename Eigen::Tensor<T, 2> weights;
    typename Eigen::Tensor<T, 2> bias;

public:
    DenseLayer() : Layer<float, 2, 2>(), weights(O, I), bias(O, 1) {
        inp()
    }
    ~DenseLayer() {}

    Eigen::Tensor<T, 2>& forward(Eigen::Tensor<T, 1>& input_tensor) {
        out = (weights * input_tensor) + bias;
        return out;
    }

    T backward(T output_gradient, T learning_rate) {
        typename T error = output_gradient * out;
        weights -= learning_rate * error;
        bias -= learning_rate * error;
        return error;
    }
};

#endif // DENSE_LAYER_HPP