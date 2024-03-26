#include "Layer.hpp"
#include "DenseLayer.hpp"
#include "ActivationLayer.hpp"
#include "ActivationFunctions.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <array>
#include <memory>
using std::cout, std::array, std::unique_ptr;



int main() {
    DenseLayer<5, 4> dl1;
    DenseLayer<4, 3> dl2;
    DenseLayer<3, 2> dl3;
    ActivationLayer<2, struct_tanh<>, tanh_prime<>> al1;
    Eigen::Matrix<float, 5, 1> input = Eigen::Matrix<float, 5, 1>::Random();
    cout << "made input:\n" << input << endl;
    auto res = dl1.forward({std::make_unique<Eigen::Matrix<float, 5, 1>>(input)});
    auto res2 = dl2.forward(std::move(res));
    auto res3 = dl3.forward(std::move(res2));

    Matrix<float, 2, 1> exgr = Eigen::Matrix<float, 2, 1>::Random();

    array<unique_ptr<Matrix<float, 2, 1>>, 1> exgrad =
        {std::make_unique<Matrix<float, 2, 1>>(exgr)};

    cout << "\n\nOutput:\n" << *res3[0] << endl;
    auto res4 = al1.forward(std::move(res3));
    cout << "\n\nRelu:\n" << *res4[0] << endl;
    auto res5 = al1.backward(std::move(exgrad), 0.1);
    cout << "\n\nPost Back ReLu:\n" << *res5[0] << endl;
    auto res6 = dl3.backward(std::move(res5), 0.1);
    cout << "\n\nPost Back Dense:\n" << *res6[0] << endl;

    
    return 0;
}

