#include "Layer.hpp"
#include "DenseLayer.hpp"
#include "ActivationLayer.hpp"
#include "ActivationFunctions.hpp"
#include "MeanSquaredError.hpp"
#include "LayerVector.hpp"
#include "LoadImage.hpp"
#include "Pipeline.hpp"
#include "Model.hpp"
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

    Model<> model(pipeline);

    Matrix<float, Dynamic, Dynamic> input1(2, 1);
    input1 << 0, 0;
    Matrix<float, Dynamic, Dynamic> input2(2, 1);
    input2 << 0, 1;
    Matrix<float, Dynamic, Dynamic> input3(2, 1);
    input3 << 1, 0;
    Matrix<float, Dynamic, Dynamic> input4(2, 1);
    input4 << 1, 1;

    
    Matrix<float, Dynamic, Dynamic> true_res1(1, 1);
    true_res1 << 0;
    Matrix<float, Dynamic, Dynamic> true_res2(1, 1);
    true_res2 << 1;


    model.add_training_data({input1}, {true_res1});
    model.add_training_data({input2}, {true_res2});
    model.add_training_data({input3}, {true_res2});
    model.add_training_data({input4}, {true_res1});

    model.run_epochs(50, 0.01, 50);

    cout << "Hekko";
    return 0;
}

