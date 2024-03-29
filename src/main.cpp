#include "Layer.hpp"
#include "DenseLayer.hpp"
#include "ActivationLayer.hpp"
#include "ActivationFunctions.hpp"
#include "MeanSquaredError.hpp"
#include "LayerVector.hpp"
#include "LoadImage.hpp"
#include "Pipeline.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <memory>
#include <random>
using std::cout, std::vector, std::unique_ptr;
using Eigen::Matrix, Eigen::Dynamic;

int main() {
    Pipeline<float> pipeline;

    pipeline.pushLayer<DenseLayer<>>(
        DenseLayer<>(2, 3)
    );

    pipeline.pushLayer<ActivationLayer<f_tanh<>, f_tanh_prime<>>>(
        ActivationLayer<f_tanh<>, f_tanh_prime<>>(1, 3, 1)
    );

    pipeline.pushLayer<DenseLayer<>>(
        DenseLayer<>(3, 5)
    );

    pipeline.pushLayer<ActivationLayer<f_tanh<>, f_tanh_prime<>>>(
        ActivationLayer<f_tanh<>, f_tanh_prime<>>(1, 5, 1)
    );

    pipeline.pushLayer<DenseLayer<>>(
        DenseLayer<> (5, 3)
    );

    pipeline.pushLayer<ActivationLayer<f_tanh<>, f_tanh_prime<>>>(
        ActivationLayer<f_tanh<>, f_tanh_prime<>>(1,3,1)
    );

    ;
    pipeline.pushLayer<DenseLayer<>>(
        DenseLayer<>(3, 1)
    );

    pipeline.pushLayer<ActivationLayer<f_tanh<>, f_tanh_prime<>, float>>(
        ActivationLayer<f_tanh<>, f_tanh_prime<>>(1,1,1)
    );
    
    pipeline.pushEndLayer<MeanSquaredError<float>>(
        MeanSquaredError<>(1,1,1)
    );

    // XOR MODEL
    vector<vector<Matrix<float, Dynamic, Dynamic>>> input;
    Matrix<float, Dynamic, Dynamic> input1(2, 1);
    input1 << 0, 0;
    input.push_back({input1});
    Matrix<float, Dynamic, Dynamic> input2(2, 1);
    input2 << 0, 1;
    input.push_back({input2});
    Matrix<float, Dynamic, Dynamic> input3(2, 1);
    input3 << 1, 0;
    input.push_back({input3});
    Matrix<float, Dynamic, Dynamic> input4(2, 1);
    input4 << 1, 1;
    input.push_back({input4});

    
    vector<vector<Matrix<float, Dynamic, Dynamic>>> true_res;
    Matrix<float, Dynamic, Dynamic> true_res1(1, 1);
    true_res1 << 0;
    true_res.push_back({true_res1});
    Matrix<float, Dynamic, Dynamic> true_res2(1, 1);
    true_res2 << 1;
    true_res.push_back({true_res2});
    true_res.push_back({true_res2});
    true_res.push_back({true_res1});

    int epochs = 200;
    for (int j = 0; j < epochs; j++) {
        for (int i = 0; i < 4; i++){
            float error = pipeline.trainPipeline(input[i], true_res[i], 0.01);
            if(j % 10 == 0){
                cout << "Error on " << j << ", " << i << ": " << error << endl;
            }
        }
    }

    {// predict
    pair<float, vector<Matrix<float, Dynamic, Dynamic>>> res1 = pipeline.testPipeline(
        input[1], true_res[1]
    );

    cout << "Error: " << res1.first << endl;
    cout << "Pred: " << res1.second[0] << endl;}
    {// predict
    pair<float, vector<Matrix<float, Dynamic, Dynamic>>> res1 = pipeline.testPipeline(
        input[2], true_res[2]
    );

    cout << "Error: " << res1.first << endl;
    cout << "Pred: " << res1.second[0] << endl;}
    {// predict
    pair<float, vector<Matrix<float, Dynamic, Dynamic>>> res1 = pipeline.testPipeline(
        input[3], true_res[3]
    );

    cout << "Error: " << res1.first << endl;
    cout << "Pred: " << res1.second[0] << endl;}
    
    return 0;
}

