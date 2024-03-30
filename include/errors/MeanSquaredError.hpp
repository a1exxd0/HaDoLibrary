#ifndef MEAN_SQUARED_ERROR_HPP
#define MEAN_SQUARED_ERROR_HPP

#include <Eigen/Dense>
#include <vector>
#include "Layer.hpp"
#include "EndLayer.hpp"
#include <memory>
#include <iostream>

using Eigen::Matrix;
using std::vector;
using Eigen::Dynamic;

/**
 * @brief Mean squared error class. T will only work for float, double,
 * or long double.
 * 
 * @tparam T Data type (float for speed, double accuracy) (optional)
*/
template<typename T=float>
class MeanSquaredError : public EndLayer<T> {
private:

    // Convenience typedef
    typedef Matrix<T, Dynamic, Dynamic> MatrixD;

    // Assert that T is either float, double, or long double at compiler time
    static_assert(
        std::is_same<T, float>::value 
        || std::is_same<T, double>::value
        || std::is_same<T, long double>::value,
        "T must be either float, double, or long double."
    );

public:

    // Default constructor
    MeanSquaredError(int D, int R, int C) : EndLayer<T>(D, R, C) {}

    // Copy constructor
    MeanSquaredError(const MeanSquaredError& mse) : EndLayer<T>(mse.getDepth(), mse.getRows(), mse.getCols()) {}

    // Clone returning unique ptr
    std::unique_ptr<EndLayer<T>> clone() const {
        return std::make_unique<MeanSquaredError>(*this);
    }

    // Destructor
    ~MeanSquaredError() {}

    /**
     * @brief Calculate mean squared error and return of type T
     * 
     * @param res Result tensor
     * @param true_res True result tensor
    */
    T forward(vector<MatrixD>& res, vector<MatrixD>& true_res) {

        // Check they must be of same size & dimensions
        if (res.size() != true_res.size()) {
            throw std::invalid_argument("Result and true result must be of same size.");
        }
        if ((res[0]).rows() != (true_res[0]).rows() || (res[0]).cols() != (true_res[0]).cols()) {
            throw std::invalid_argument("Result and true result must be of same dimensions.");
        }

        // Calculate statistical mean square sum of all matrices in tensor
        T error = 0;
        for (int i = 0; i < this->D; i++) {
            error += (res[i].array() - true_res[i].array()).square().sum();
        }
        return (error / ((this->D)*(this->R)*(this->C)));
    }

    /**
     * @brief Get derivative of error w.r.t mean squared error for every element in result.
     * 
     * @param res Result vector from model
     * @param true_res True value (label for data)
    */
    vector<MatrixD> backward(vector<MatrixD>& res, vector<MatrixD>& true_res) {

        // Check they must be of same size & dimensions
        if (res.size() != true_res.size()) {
            throw std::invalid_argument("Result and true result must be of same size.");
        }
        if ((res[0]).rows() != (true_res[0]).rows() || (res[0]).cols() != (true_res[0]).cols()) {
            throw std::invalid_argument("Result and true result must be of same dimensions.");
        }

        vector<MatrixD> grad(this->D);

        // For each layer of depth in tensor
        for (int i = 0; i < this->D; i++){

            // Calculate gradient (derivative of mse)
            grad[i] = 2*(res[i].array() - true_res[i].array());
        }

        // Return gradient matrix
        return (grad);
    }
};


#endif // MEAN_SQUARED_ERROR_HPP