#include "DeepNeuralNetwork.hpp"
#include "XorModel.cpp"
using std::cout, std::vector, std::unique_ptr;
using Eigen::Matrix, Eigen::Dynamic;

int main() {
    
    // Run test model
    DNNExample::xorModel();
    
    return 0;
}

