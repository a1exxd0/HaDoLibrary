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

namespace hado {

/**
 * @brief End layer class for getting error gradient w.r.t results
 * 
 * @tparam T Data type (float for speed, double accuracy) (optional)
*/
template<typename T=float>
class EndLayer {
private:

    // Assert that T is either float, double, or long double at compiler time
    static_assert(
        std::is_same<T, float>::value 
        || std::is_same<T, double>::value
        || std::is_same<T, long double>::value,
        "T must be either float, double, or long double."
    );

protected:

    // Convenience typedef
    typedef Matrix<T, Dynamic, Dynamic> MatrixD;

    // Enforce that these are constructed
    int D, R, C;

    /**
     * @brief Constructor for basic end layer of neural network
     * 
     * @param D Depth of end node
     * @param R Number of rows of end node
     * @param C Number of columns of end node
    */
    EndLayer(int D, int R, int C) {
        this->D = D;
        this->R = R;
        this->C = C;
    }

public:

    // Getters
    int getDepth() const { return D; }
    int getRows() const { return R; }
    int getCols() const { return C; }

    // Copy constructor
    EndLayer(const EndLayer& el) {
        D = el.getDepth();
        R = el.getRows();
        C = el.getCols();
    }

    // Virtual clone
    virtual std::unique_ptr<EndLayer<T>> clone() const = 0;

    // Virtual destructor
    virtual ~EndLayer(){}

    // Calculate error w.r.t results
    virtual T forward(
        vector<MatrixD>& res, vector<MatrixD>& true_res) = 0;

    // Calculate error gradient w.r.t results
    virtual vector<MatrixD> backward(
        vector<MatrixD>& res, vector<MatrixD>& true_res) = 0;
};

}

#endif // END_LAYER_HPP
