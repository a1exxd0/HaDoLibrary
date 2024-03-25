#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Dense>
#include <array>
#include <memory>

using Eigen::Matrix;
using std::array;

/**
 * @brief Base layer class. T will only work for float, double,
 * or long double.
 * 
 * @details operator[] Overloaded operator to access output matrix. Matrices
 * stored in an array, so access by index.
 * 
 * @tparam T Data type (float for speed, double accuracy)
 * @tparam I Input tensor depth
 * @tparam O Output tensor depth
 * @tparam RI Rows in input tensor
 * @tparam CI Columns in input tensor
 * @tparam RO Rows in output tensor
 * @tparam CO Columns in output tensor
*/
template<typename T, int I, int O, int RI, int CI, int RO, int CO>
class Layer {
private:

    // Assert that T is either float, double, or long double at compiler time
    #pragma GCC diagnostic ignored "-Wparentheses"
    static_assert(
        std::is_same<T, float>::value 
        || std::is_same<T, double>::value
        || std::is_same<T, long double>::value
        && (I > 0 && O > 0 && RI > 0 && CI > 0 && RO > 0 && CO > 0),
        "T must be either float, double, or long double.\nRest must be positive integers."
    );

public:

    // Input tensor
    array<std::unique_ptr<Matrix<T, RI, CI>>, I> inp;

    // Output tensor
    array<std::unique_ptr<Matrix<T, RO, CO>>, O> out;
    
    // Default constructor
    Layer() : inp(), out() {}

    // Forward propagation
    virtual array
        <std::unique_ptr<Matrix<T, RO, CO>>, O> forward(
            array<std::unique_ptr<Matrix<T, RI, CI>>, I> input_tensor) = 0;

    virtual T backward(T output_gradient, T learning_rate) = 0;

};

#endif // LAYER_HPP