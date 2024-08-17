#ifndef CROSS_ENTROPY_LOSS_HPP
#define CROSS_ENTROPY_LOSS_HPP

#include <Eigen/Dense>
#include <vector>
#include "HaDo/base/EndLayer.hpp"
#include <memory>
#include <iostream>

using Eigen::Matrix;
using std::vector;
using Eigen::Dynamic;

namespace hado {

/**
 * @brief CrossEntropyLoss class - acts as a loss function for the neural network
 * 
 * @tparam T scalar type (float, double, long double)
*/
template<typename T=float>
class CrossEntropyLoss : public EndLayer<T> {
private:

    // Convenience typedef
    using typename EndLayer<T>::MatrixD;

public:

    /**
     * @brief Construct a new Cross Entropy Loss object.
     * This is the final end layer of a classification network and calculates loss
     * after passing through a softmax function. Any other usage will invalidate the
     * function's assumptions.
     * 
     * @param R number of classes/rows in output
    */
    explicit CrossEntropyLoss(int R) : EndLayer<T>(1, R, 1) {
        if (R < 2){
            std::cerr << "Must be a classification of 2 outputs minimum." << endl;
            assert(R >= 2);
        }
    }

    // Copy constructor
    CrossEntropyLoss(const CrossEntropyLoss& cel) : EndLayer<T>(cel) {}

    // Clone
    unique_ptr<EndLayer<T>> clone() const override {
        return std::make_unique<CrossEntropyLoss<T>>(*this);
    }

    // Destructor
    ~CrossEntropyLoss() override = default;

    /**
     * @brief Forward pass of the CrossEntropyLoss layer. True res must be a single
     * vertical vector of depth 1 with 1 entry of value one and the rest 0.
     * 
     * @param res vector of matrices containing the output of the previous layer
     * @param true_res vector of matrices containing the true output of the network
     * @return T loss for that input
    */
    T forward(vector<MatrixD>& res, vector<MatrixD>& true_res) override {
        // Assert tensor dimensions
        this->assertInputDimensions(res);
        this->assertInputDimensions(true_res);

        T loss = 0;

        // Essentially the difference between true_res and the log of res (cross entropy loss definition)
        // Sum for total error and * -1 to make it positive
        loss = -1 * (true_res[0].array() * res[0].array().log()).array().sum();
        return loss;
    }

    /**
     * @brief Backward pass of the CrossEntropyLoss layer
     * 
     * @param res vector of matrices containing the output of the previous layer
     * @param true_res vector of matrices containing the true output of the network
     * @return vector<MatrixD> gradient of the loss with respect to the input
    */
    vector<MatrixD> backward(vector<MatrixD>& res, vector<MatrixD>& true_res) override {
        vector<MatrixD> grad;

        // Provable derivative of loss w.r.t inputs into CrossEntropyLoss
        grad.push_back(res[0].array()-true_res[0].array());
        return grad;
    }

};

}

#endif
