#ifndef CROSS_ENTROPY_LOSS_HPP
#define CROSS_ENTROPY_LOSS_HPP

#include <Eigen/Dense>
#include <vector>
#include "Layer.hpp"
#include "EndLayer.hpp"
#include <memory>
#include <iostream>

using Eigen::Matrix;
using std::vector;
using Eigen::Dynamic;

template<typename T=float>
class CrossEntropyLoss : public EndLayer<T> {
private:

    // Convenience typedef
    typedef Matrix<T, Dynamic, Dynamic> MatrixD;

    static_assert(
        std::is_same<T, float>::value 
        || std::is_same<T, double>::value
        || std::is_same<T, long double>::value,
        "T must be either float, double, or long double."
    );

public:

    // Constructor
    CrossEntropyLoss(int R) : EndLayer<T>(1, R, 1) {}

    // Copy constructor
    CrossEntropyLoss(const CrossEntropyLoss& cel) : EndLayer<T>(cel) {}

    // Clone
    unique_ptr<EndLayer<T>> clone() const {
        return std::make_unique<CrossEntropyLoss<T>>(*this);
    }

    // Destructor
    ~CrossEntropyLoss() {}

    // Forward
    T forward(vector<MatrixD>& res, vector<MatrixD>& true_res){
        T loss = 0;
        loss = -1 * (true_res[0].array() * res[0].array().log()).array().sum();
        return loss;
    }

    // Backward
    vector<MatrixD> backward(vector<MatrixD>& res, vector<MatrixD>& true_res){
        vector<MatrixD> grad;
        grad.push_back(res[0].array()-true_res[0].array());
        return grad;
    }

};



#endif