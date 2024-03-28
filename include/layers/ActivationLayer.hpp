#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

#include <Eigen/Dense>
#include <vector>
#include "Layer.hpp"
#include <memory>
#include <iostream>

using Eigen::Matrix;
using std::vector;
using std::cout;
using Eigen::Dynamic;
using std::unique_ptr;

/**
 * @brief Activation layer class.
 * 
 * @details Activation layer class that applies an activation function
 * 
 * @tparam D Depth of input tensor
 * @tparam R Rows in input tensor
 * @tparam C Columns in input tensor
 * @tparam Activation Activation function
 * @tparam ActivationPrime Derivative of activation function
 * @tparam T Data type (float for speed, double accuracy) (optional)
*/
template<typename Activation, typename ActivationPrime, typename T=float>
class ActivationLayer : public Layer<T> {
private:

    // Convenience typedef
    typedef Matrix<T, Dynamic, Dynamic> MatrixD;

    int D, R, C;

    // Assert that T is either float, double, or long double at compiler time
    static_assert(
        std::is_same<T, float>::value 
        || std::is_same<T, double>::value
        || std::is_same<T, long double>::value,
        "T must be either float, double, or long double."
    );

    // Assert that Activation and ActivationPrime are functions that take a scalar and return a scalar
    static_assert(
        std::is_invocable_r<T, Activation, T>::value,
        "Activation must be a function that takes a scalar and returns a scalar."
    );
    static_assert(
        std::is_invocable_r<T, ActivationPrime, T>::value,
        "ActivationPrime must be a function that takes a scalar and returns a scalar."
    );

public:

    /**
     * @brief Construct a new Activation Layer object
    */
    ActivationLayer(int D, int R, int C) : Layer<T>(D, D, R, C, R, C) {
        this->D = D;
        this->R = R;
        this->C = C;
        this->inp = vector<MatrixD>(D);
        this->out = vector<MatrixD>(D);
    }

    // Copy constructor
    ActivationLayer(const ActivationLayer<Activation, ActivationPrime, T>& other) : Layer<T>(other.D, other.D, other.R, other.C, other.R, other.C) {
        this->D = other.D;
        this->R = other.R;
        this->C = other.C;
        this->inp = other.inp;
        this->out = other.out;
    }

    // Destructor
    ~ActivationLayer() {}

    /**
     * @brief Forward pass of the activation layer.
     * 
     * @param input_tensor Input tensor
     * @return Output tensor of same dimensions as input tensor
    */
    vector<MatrixD> forward(vector<MatrixD>& input_tensor) {
        if (input_tensor.size() != (size_t) D || input_tensor[0].rows() != R || input_tensor[0].cols() != C) {
            throw std::invalid_argument("Input tensor must have depth D.");
        }

        // Get copy because we need to pass one forward, and one stays in layer
        vector<MatrixD> out_copy(D);

        // Iterate through depth of tensor
        for (int i = 0; i < D; i++){

            // Move input matrix into layer attribute inp
            this->inp[i] = input_tensor[i];

            // Calculate output tensor for single layer
            auto res = this->inp[i].unaryExpr(Activation());

            // Copy output tensor to return
            out_copy[i] = res;

            // Store a copy of the output tensor in layer attribute out
            this->out[i] = res;
        }

        return (out_copy);
    }

    /**
     * @brief Backward pass of the activation layer. Output gradient tensor must be a size 1 std::vector
     * of std::unique_ptr<MatrixD>. Input gradient tensor (returned) is the same size.
     * 
     * @param output_gradient Output gradient tensor (one dimensional, must have right size)
     * @param learning_rate Learning rate
     * @return vector<std::unique_ptr<MatrixD>> Input gradient tensor
     */
    #pragma GCC diagnostic ignored "-Wunused-parameter"
    #pragma GCC optimize("O3")
    vector<MatrixD> backward(vector<MatrixD>& output_gradient, T learning_rate) {

        // Assert that output gradient tensor is the same size as the input tensor
        if (output_gradient.size() != (size_t) D || output_gradient[0].rows() != R || output_gradient[0].cols() != C) {
            throw std::invalid_argument("Output gradient tensor must have depth D.");
        }

        // Array to store input gradient (not the input)
        vector<MatrixD> input_gradient(D);

        // Iterate through depth of tensor
        for (int i = 0; i < D; i++){

            // Calculate input gradient for single layer
            auto res = 
                this->out[i].unaryExpr(ActivationPrime())
                    .cwiseProduct(output_gradient[i]);

            // Store input gradient
            input_gradient[i] = res;
        }

        return {input_gradient};
    }

};

#endif