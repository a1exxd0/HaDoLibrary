#ifndef END_LAYER_HPP
#define END_LAYER_HPP

#include <Eigen/Dense>
#include <array>
#include "Layer.hpp"
#include <memory>
#include <iostream>

using Eigen::Matrix;
using std::array;

/**
 * @brief End layer class for getting error gradient w.r.t results
 * 
 * @tparam D Depth of input tensor
 * @tparam R Rows in input tensor
 * @tparam C Columns in input tensor
 * @tparam T Data type (float for speed, double accuracy) (optional)
*/
template<int D, int R, int C, typename T=float>
class EndLayer {
private:

    // Assert that T is either float, double, or long double at compiler time
    static_assert(
        std::is_same<T, float>::value 
        || std::is_same<T, double>::value
        || std::is_same<T, long double>::value,
        "T must be either float, double, or long double."
    );

public:

    // Default constructor
    EndLayer() {}

    // Destructor
    ~EndLayer() {}

    // Calculate error w.r.t results
    virtual T forward(
        array<unique_ptr<Matrix<T, R, C>>, D> res, array<unique_ptr<Matrix<T, R, C>>, D> true_res) = 0;

    // Calculate error gradient w.r.t results
    virtual array<unique_ptr<Matrix<T, R, C>>, D> backward(
        array<unique_ptr<Matrix<T, R, C>>, D> res, array<unique_ptr<Matrix<T, R, C>>, D> true_res) = 0;
};

#endif // END_LAYER_HPP