#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Dense>

/**
 * @brief Base layer class. T will only work for float, double,
 * or long double.
 * 
 * @tparam T Data type (float for speed, double accuracy)
 * @tparam I Input tensor dimension
 * @tparam O Output tensor dimension
*/
template<typename T, int I, int O>
class Layer {
private:

    // Assert that T is either float, double, or long double at compiler time
    static_assert(
        std::is_same<T, float>::value 
        || std::is_same<T, double>::value
        || std::is_same<T, long double>::value,
        "T (first template param) must be either float, double, or long double."
    );

public:

    // Pure virtual functions for forward and backward propagation
    virtual Eigen::Tensor<T, O>& forward(Eigen::Tensor<T, I>& input_tensor) = 0;
    virtual T backward(T output_gradient, T learning_rate) = 0;
};

#endif // LAYER_HPP