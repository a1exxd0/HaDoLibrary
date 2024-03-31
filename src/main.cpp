#include "DeepNeuralNetwork.hpp"
#include "XorModel.cpp"
// #include "ConvolutionalLayer.hpp"
using std::cout, std::vector, std::unique_ptr;
using Eigen::Matrix, Eigen::Dynamic;

int main() {
    
    // Run test model
    // DNNExample::xorModel();

    // 3d test

    Pipeline<double> pipeline;
    pipeline.pushLayer(
        ActivationLayer<f_tanh<double>, f_tanh_prime<double>, double>(5,50, 50)
    );
    pipeline.pushLayer(
        ActivationLayer<sigmoid<double>, sigmoid_prime<double>, double>(5,50,50)
    );
    pipeline.pushLayer(
        ActivationLayer<f_tanh<double>, f_tanh_prime<double>, double>(5,50,50)
    );
    pipeline.pushEndLayer(
        MeanSquaredError<double>(5,50,50)
    );

    // add data
    vector<Matrix<double, Dynamic, Dynamic>> data;
    for (int i = 0; i < 5; i++) {
        data.push_back(Matrix<double, Dynamic, Dynamic>::Random(50,50));
    }

    Model<double> model(pipeline);

    model.add_training_data(data, data);

    model.run_epochs(30000, 0.01, 10);
    return 0;
}

