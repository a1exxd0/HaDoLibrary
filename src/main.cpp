#include "Layer.hpp"
#include "DenseLayer.hpp"
#include "ActivationLayer.hpp"
#include "ActivationFunctions.hpp"
#include "MeanSquaredError.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <array>
#include <memory>
using std::cout, std::array, std::unique_ptr;



int main() {
    DenseLayer<5, 4> dl1;
    DenseLayer<4, 3> dl2;
    DenseLayer<3, 2> dl3;
    ActivationLayer<1, 2, 1, f_tanh<>, f_tanh_prime<>> al1;
    MeanSquaredError<1, 2, 1> mse;
    Eigen::Matrix<float, 5, 1> input = Eigen::Matrix<float, 5, 1>::Random();
    cout << "made input:\n" << input << endl;
    auto res = dl1.forward({std::make_unique<Eigen::Matrix<float, 5, 1>>(input)});
    auto res2 = dl2.forward(std::move(res));
    auto res3 = dl3.forward(std::move(res2));

    Matrix<float, 2, 1> exgr = Eigen::Matrix<float, 2, 1>::Random();

    array<unique_ptr<Matrix<float, 2, 1>>, 1> true_vals =
        {std::make_unique<Matrix<float, 2, 1>>(exgr)};

    cout << "\n\nCorrect:\n" << *true_vals[0] << endl;

    cout << "\n\nOutput:\n" << *res3[0] << endl;
    auto res4 = al1.forward(std::move(res3));
    cout << "\n\nRelu:\n" << *res4[0] << endl;

    auto backgrad = mse.backward(std::move(res4), std::move(true_vals));
    
    cout << "\n\nBackward:\n" << *backgrad[0] << endl;


    Eigen::Matrix<float, 3, 3> input2 = Eigen::Matrix<float, 3, 3>::Random();
    cout << "made input:\n" << input2 << endl;
    auto uptr = std::make_unique<Eigen::Matrix<float, 3, 3>>(input2);
    ActivationLayer<1, 3, 3, relu<>, relu_prime<>> al2;
    auto pass = al2.forward({std::move(uptr)});

    //cout << "\n\nRelu:\n" << *pass[0] << endl;
    
    return 0;
}

