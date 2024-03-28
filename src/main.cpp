#include "Layer.hpp"
#include "DenseLayer.hpp"
#include "ActivationLayer.hpp"
#include "ActivationFunctions.hpp"
#include "MeanSquaredError.hpp"
#include "LayerVector.hpp"
#include "LoadImage.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <memory>
#include <random>
using std::cout, std::vector, std::unique_ptr;
using Eigen::Matrix, Eigen::Dynamic;

int main() {
    // Seed the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Create a random number generator using a normal distribution
    std::normal_distribution<float> distribution(0.0, 1.0);

    // Generate random numbers with Eigen
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> expected;
    expected.resize(2, 1);
    expected << 0, 1;


    LayerVector<float> lv;
    DenseLayer<float> dl1(2, 3);
    lv.pushLayer<DenseLayer<float>>(dl1);
    ActivationLayer<f_tanh<>, f_tanh_prime<>, float> al1(1, 3, 1);
    lv.pushLayer<ActivationLayer<f_tanh<>, f_tanh_prime<>, float>>(al1);
    DenseLayer<float> dl2(3, 1);
    lv.pushLayer<DenseLayer<float>>(dl2);
    ActivationLayer<f_tanh<>, f_tanh_prime<>, float> al2(1, 1, 1);
    lv.pushLayer<ActivationLayer<f_tanh<>, f_tanh_prime<>, float>>(al2);
    

    MeanSquaredError<float> mse(1, 1, 1);

    // XOR MODEL
    vector<Matrix<float, Dynamic, Dynamic>> input;
    Matrix<float, Dynamic, Dynamic> input1(2, 1);
    input1 << 0, 0;
    input.push_back(input1);
    Matrix<float, Dynamic, Dynamic> input2(2, 1);
    input2 << 0, 1;
    input.push_back(input2);
    Matrix<float, Dynamic, Dynamic> input3(2, 1);
    input3 << 1, 0;
    input.push_back(input3);
    Matrix<float, Dynamic, Dynamic> input4(2, 1);
    input4 << 1, 1;
    input.push_back(input4);

    
    vector<Matrix<float, Dynamic, Dynamic>> true_res;
    Matrix<float, Dynamic, Dynamic> true_res1(1, 1);
    true_res1 << 0;
    true_res.push_back(true_res1);
    Matrix<float, Dynamic, Dynamic> true_res2(1, 1);
    true_res2 << 1;
    true_res.push_back(true_res2);
    true_res.push_back(true_res2);
    true_res.push_back(true_res1);

    int epochs = 3000;
    for (int j = 0; j < epochs; j++) {
        for (int i = 0; i < 4; i++){
            auto out = lv.forward({input[i]});
            vector<Matrix<float, Dynamic, Dynamic>> x = {true_res[i]};
            float error = mse.forward(out, x);
            if(j % 300 == 0){
                cout << "Error on " << j << ", " << i << ": " << error << endl;
            }
            lv.backward(mse.backward(out, x), 0.01);
        }
    }

    auto out = lv.forward({input[0]});
    vector<Matrix<float, Dynamic, Dynamic>> x = {true_res1};
    float error = mse.forward(out, x);
    cout << "0, 0 yields: " << out[0] << " with error" << error << endl;

    auto out2 = lv.forward({input[1]});
    vector<Matrix<float, Dynamic, Dynamic>> x2 = {true_res2};
    float error2 = mse.forward(out2, x2);
    cout << "0, 1 yields: " << out2[0] << " with error" << error2 << endl;


    
    //cout << "\n\nloss:\n" << loss << endl;
    
    return 0;
}

