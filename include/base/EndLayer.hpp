#ifndef END_LAYER_HPP
#define END_LAYER_HPP

#include <Eigen/Dense>
#include <vector>
#include "Layer.hpp"
#include <memory>
#include <iostream>

using Eigen::Matrix;
using std::vector;
using Eigen::Dynamic;

/**
 * @brief End layer class for getting error gradient w.r.t results
 * 
 * @tparam T Data type (float for speed, double accuracy) (optional)
*/
template<typename T=float>
class EndLayer {
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

protected:

    // Enforce that this be used
    int D, R, C;

    // Default constructor
    EndLayer(int D, int R, int C) {
        this->D = D;
        this->R = R;
        this->C = C;
    }

public:

    // Getters
    int getDepth() { return D; }
    int getRows() { return R; }
    int getCols() { return C; }

    // Calculate error w.r.t results
    virtual T forward(
        vector<MatrixD>& res, vector<MatrixD>& true_res) = 0;

    // Calculate error gradient w.r.t results
    virtual vector<MatrixD> backward(
        vector<MatrixD>& res, vector<MatrixD>& true_res) = 0;
};

#endif // END_LAYER_HPP