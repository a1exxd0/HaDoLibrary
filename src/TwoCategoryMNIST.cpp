#include <HaDo/ConvolutionalNeuralNetwork>

using namespace hado;

void TwoCategoryMNIST(){
    Pipeline<double> pipeline;

    pipeline.pushLayer(
        ConvolutionalLayer<double, relu<double>, relu_prime<double>>(1, 2, 28, 28, 3, 1, 0)
    );

    pipeline.pushLayer(
        MaxPoolLayer<double>(2, 26, 26, 2, 2, 0)
    );

    pipeline.pushLayer(
        ConvolutionalLayer<double, relu<double>, relu_prime<double>>(2, 1, 13, 13, 4, 1, 0)
    );
    
    pipeline.pushLayer(
        MaxPoolLayer<double>(2, 9, 9, 3, 3, 0)
    );

}