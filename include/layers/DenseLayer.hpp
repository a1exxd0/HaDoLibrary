#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include <Eigen/Dense>
#include <vector>
#include "Layer.hpp"
#include <memory>
#include <iostream>
#include <chrono>
#include <random>

using Eigen::Matrix;
using std::vector;
using std::cout;
using Eigen::Dynamic;
using std::unique_ptr;

/**
 * @brief Dense layer class. T will only work for float, double,
 * or long double.
 * 
 * @tparam T Data type (float for speed, double accuracy) (optional)
*/
template<typename T=float> 
class DenseLayer : public Layer<T>{
private:

    // Convenience typedef
    typedef Matrix<T, Dynamic, Dynamic> MatrixD;

    // Convenience private variables
    int I, O;

    // Check if T is float, double, or long double
    static_assert(
        std::is_same<T, float>::value 
        || std::is_same<T, double>::value
        || std::is_same<T, long double>::value,
        "T (first template param) must be either float, double, or long double."
    );

    // Weights and bias tensors
    MatrixD weights;
    MatrixD bias;

public:

    /**
     * @brief Construct a new Dense Layer object
     * 
     * Initializes weights and bias with random values for this layer.
     * 
     * @param 
    */
    DenseLayer(int I, int O) : Layer<T>(1, 1, I, 1, O, 1){

        // Get the current time as a seed
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        
        // Seed the random number generator
        std::mt19937 rng(seed);

        // Use the seeded RNG with Eigen::Random
        Eigen::MatrixXf matrix(3, 3);
        Eigen::Random<std::mt19937> random(rng);
        matrix = random.normal();

        this->weights = MatrixD(O, I); this->weights = random.normal()
        this->bias = MatrixD(O, 1);
        this->inp = vector<MatrixD>(1);
        this->out = vector<MatrixD>(1);
        this->I = I;
        this->O = O;
    }

    // Copy constructor
    DenseLayer(const DenseLayer<T>& other) : Layer<T>(1, 1, other.I, 1, other.O, 1){
        this->weights = other.weights;
        this->bias = other.bias;
        this->inp = other.inp;
        this->out = other.out;
        this->I = other.I;
        this->O = other.O;
    }

    // Destructor
    ~DenseLayer() {}

    /**
     * @brief Forward pass of the dense layer. Input tensor must be a size 1 vector
     * of dimensions I x 1.
     * 
     * @param input_tensor Input tensor (one dimensional, must have right size)
     * @return vector<unique_ptr<MatrixD>> Output tensor
    */
    vector<MatrixD> forward(vector<MatrixD>& input_tensor) {

        // Validity check
        if (input_tensor.size() != 1 || (input_tensor[0]).cols() != 1 || (input_tensor[0]).rows() != I){
            cout << "Input tensor must be a size 1 vector of dimensions I x 1." << endl;
            exit(1);
        }

        // Move input tensor into layer attribute inp
        this->inp = input_tensor;

        // Calculate output tensor
        auto res = (weights * (this->inp[0]) + bias);

        // Copy output tensor to return
        vector<MatrixD> out_copy;

        // Make a unique pointer for the copy
        out_copy.push_back(res);

        // Store a copy of the output tensor in layer attribute out
        this->out[0] = res;

        // Return output tensor
        return (out_copy); 
    }

    /**
     * @brief Backward pass of the dense layer. Output gradient tensor must be a size 1 std::array
     * of std::unique_ptr<Matrix<T, O, 1>>.
     * 
     * @param output_gradient Output gradient tensor (one dimensional, must have right size)
     * @param learning_rate Learning rate for gradient descent (0 < learning_rate < 1)
     * @return array<std::unique_ptr<Matrix<T, I, 1>>, 1> Input gradient tensor
    */
    vector<MatrixD> backward(vector<MatrixD>& output_gradient, T learning_rate) {

        // Validity check
        if (output_gradient.size() != 1 || (output_gradient[0]).rows() != O || (output_gradient[0]).cols() != 1){
            cout << "Output gradient tensor must be a size 1 vector of dimensions O x 1." << endl;
            exit(1);
        }

        // Calculate weight gradient, bias gradient, and input gradient
        auto weight_gradient = (output_gradient[0]) * (this->inp[0]).transpose();
        auto bias_gradient = output_gradient[0];
        auto input_gradient = this->weights.transpose() * (output_gradient[0]);

        // Update weights and bias
        this->weights -= learning_rate * weight_gradient;
        this->bias -= learning_rate * bias_gradient;

        // Return input gradient
        return {input_gradient};
    }
};

#endif // DENSE_LAYER_HPP