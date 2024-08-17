#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

#include <Eigen/Dense>
#include <vector>
#include "HaDo/base/Layer.hpp"
#include <memory>
#include <thread>
#include <iostream>

using Eigen::Matrix;
using std::vector;
using std::cout;
using Eigen::Dynamic;
using std::unique_ptr;

namespace hado {

/**
 * @brief Activation layer class.
 * 
 * @details Activation layer class that applies an activation function
 * 
 * @tparam Activation Activation function
 * @tparam ActivationPrime Derivative of activation function
 * @tparam T Data type (float for speed, double accuracy) (optional)
*/
template<typename Activation, typename ActivationPrime, typename T=float>
class ActivationLayer : public Layer<T> {
private:

    // Convenience typedef
    using typename Layer<T>::MatrixD;

    // Convenience variables inaccessible from outside
    int D, R, C;
    int prod;

    // Assert that Activation and ActivationPrime are functions that take a scalar and return a scalar
    static_assert(
        std::is_invocable_r_v<T, Activation, T>,
        "Activation must be a functor that takes a scalar and returns a scalar."
    );
    static_assert(
        std::is_invocable_r_v<T, ActivationPrime, T>,
        "ActivationPrime must be a functor that takes a scalar and returns a scalar."
    );

public:

    /**
     * @brief Construct a new Activation Layer object. Input and output
     * tensors are same dimesions.
     * 
     * @param D Depth of input/output tensor
     * @param R Rows in input/output tensor
     * @param C Columns in input/output tensor
    */
    ActivationLayer(int D, int R, int C) : Layer<T>(D, D, R, C, R, C) {

        this->D = D;
        this->R = R;
        this->C = C;
        this->prod = R * C;
        this->inp = vector<MatrixD>(D);
        this->out = vector<MatrixD>(D);
    }

    // Copy constructor
    ActivationLayer(const ActivationLayer<Activation, ActivationPrime, T>& other) 
        : Layer<T>(other.getInputDepth(), other.getOutputDepth(), 
            other.getInputRows(), other.getInputCols(), 
            other.getOutputRows(), other.getOutputCols()) {

            this->D = other.getInputDepth();
            this->R = other.getInputRows();
            this->C = other.getInputCols();
            this->prod = other.getInputRows() * other.getInputCols();
            this->inp = other.inp;
            this->out = other.out;
        }

    // Clone returning unique pointer
    virtual unique_ptr<Layer<T>> clone() const override {
        return std::make_unique<ActivationLayer<Activation, ActivationPrime, T>>(*this);
    }

    // Destructor
    ~ActivationLayer() override {}

    /**
     * @brief Forward pass of the activation layer.
     * 
     * @param input_tensor Input tensor
     * @return Output tensor of same dimensions as input tensor
    */
    #pragma GCC push_options
    #pragma GCC optimize("O2")
    virtual vector<MatrixD> forward(vector<MatrixD>& input_tensor) override {

        // Assert input tensor dimensions
        this->assertInputDimensions(input_tensor);

        // Get copy because we need to pass one forward, and one stays in layer
        vector<MatrixD> out_copy(D);
        #ifdef _OPENMP
            #include <omp.h>
            if (D > _MAX_DEPTH_UNTIL_THREADING && prod >= _MAX_PROD_UNTIL_THREADING){
                omp_set_num_threads(D);
                #pragma omp parallel for
                for (int i = 0; i < D; i++){
                    forward_function(input_tensor[i], this->out[i], out_copy[i]);
                }
            } else{
                // Iterate through depth of tensor
                for (int i = 0; i < D; i++){
                    forward_function(input_tensor[i], this->out[i], out_copy[i]);
                }
            }
        #else
            for (int i = 0; i < D; i++){
                forward_function(input_tensor[i], this->out[i], out_copy[i]);
            }
        #endif

        return (out_copy);
    }
    #pragma GCC pop_options
    

    /**
     * @brief Backward pass of the activation layer. Output gradient tensor must be a size 1 std::vector
     * of MatrixD. Input gradient tensor (returned) is the same size.
     * 
     * @param output_gradient Output gradient tensor (one dimensional, must have right size)
     * @param learning_rate Learning rate
     * @return vector<MatrixD> Input gradient tensor
     */
    #pragma GCC diagnostic ignored "-Wunused-parameter"
    #pragma GCC push_options
    #pragma GCC optimize("O2")
    virtual vector<MatrixD> backward(vector<MatrixD>& output_gradient, const T learning_rate) override{

        // Assert that output gradient tensor is the same size as the input tensor
        this->assertOutputDimensions(output_gradient);

        // Array to store input gradient (not the input)
        vector<MatrixD> input_gradient(D);

        #ifdef _OPENMP
            #include <omp.h>
            if (D > _MAX_DEPTH_UNTIL_THREADING && prod >= _MAX_PROD_UNTIL_THREADING){
                omp_set_num_threads(D);
                #pragma omp parallel for
                for (int i = 0; i < D; i++){
                    backward_function(output_gradient[i], this->out[i], input_gradient[i]);
                }
            } else{
                // Iterate through depth of tensor
                for (int i = 0; i < D; i++){
                    backward_function(output_gradient[i], this->out[i], input_gradient[i]);
                }
            }
        #else
            // Iterate through depth of tensor
            for (int i = 0; i < D; i++){
                backward_function(output_gradient[i], this->out[i], input_gradient[i]);
            }
        #endif

        return {input_gradient};
    }
    #pragma GCC pop_options

private:

    // Private lambda for forward pass with threading
    static constexpr auto forward_function = [](MatrixD& input, MatrixD& output, MatrixD& output_copy){

        // Apply activation function to all input elements
        output = input.unaryExpr(Activation());

        // Copy output to output_copy
        output_copy = output;
    };

    // Private lambda for backward pass with threading
    static constexpr auto backward_function = [](MatrixD& output_gradient, MatrixD& output, MatrixD& input_gradient){
        
        // Calculate input gradient for single layer
        input_gradient = output.unaryExpr(ActivationPrime())
            .cwiseProduct(output_gradient);
    };

};

}

#endif