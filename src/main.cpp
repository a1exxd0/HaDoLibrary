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
        ActivationLayer<f_tanh<double>, f_tanh_prime<double>, double>(10,100, 100)
    );
    pipeline.pushLayer(
        ActivationLayer<sigmoid<double>, sigmoid_prime<double>, double>(10,100,100)
    );
    pipeline.pushLayer(
        ActivationLayer<f_tanh<double>, f_tanh_prime<double>, double>(10,100,100)
    );
    pipeline.pushEndLayer(
        MeanSquaredError<double>(10,100,100)
    );

    // add data
    vector<Matrix<double, Dynamic, Dynamic>> data;
    for (int i = 0; i < 10; i++) {
        data.push_back(Matrix<double, Dynamic, Dynamic>::Random(100,100));
    }

    Model<double> model(pipeline);

    model.add_training_data(data, data);

    model.run_epochs(1000, 0.01, 10);
    return 0;
}

