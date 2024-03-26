#ifndef MEAN_SQUARED_ERROR_HPP
#define MEAN_SQUARED_ERROR_HPP

#include <Eigen/Dense>
#include <array>
#include "Layer.hpp"
#include "EndLayer.hpp"
#include <memory>
#include <iostream>

using Eigen::Matrix;
using std::array;
using std::unique_ptr;

/**
 * @brief Mean squared error class. T will only work for float, double,
 * or long double.
 * 
 * @tparam D Depth of input tensor
 * @tparam R Rows in input tensor
 * @tparam C Columns in input tensor
 * @tparam T Data type (float for speed, double accuracy) (optional)
*/
template<int D, int R, int C, typename T=float>
class MeanSquaredError : public EndLayer<D, R, C, T> {
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
    MeanSquaredError() : EndLayer<D, R, C, T>() {}

    // Destructor
    ~MeanSquaredError() {}

    /**
     * @brief Calculate mean squared error and return of type T
     * 
     * @param res Result tensor
     * @param true_res True result tensor
    */
    T forward(array<unique_ptr<Matrix<T, R, C>>, D> res, array<unique_ptr<Matrix<T, R, C>>, D> true_res) {
        T error = 0;

        // For each layer of depth
        for (int i = 0; i < D; i++) {

            // Sum of squared differences
            error += (res[i]->array() - true_res[i]->array()).square().sum();
        }

        // Return mean squared error
        return (error / (D*R*C));
    }

    array<unique_ptr<Matrix<T, R, C>>, D> backward(
        array<unique_ptr<Matrix<T, R, C>>, D> res, array<unique_ptr<Matrix<T, R, C>>, D> true_res) {
            array<unique_ptr<Matrix<T, R, C>>, D> grad;

            // For each layer of depth in tensor
            for (int i = 0; i < D; i++){

                // Calculate gradient (derivative of mse)
                grad[i] = std::make_unique<Matrix<T, R, C>>(
                    2*(res[i]->array() - true_res[i]->array()));
            }

            // Return gradient matrix
            return (grad);
        }
};


#endif // MEAN_SQUARED_ERROR_HPP