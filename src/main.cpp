#include "DeepNeuralNetwork.hpp"
#include "XorModel.cpp"
// #include "ConvolutionalLayer.hpp"
#include "MaxPoolLayer.hpp"
using std::cout, std::vector, std::unique_ptr;
using Eigen::Matrix, Eigen::Dynamic;

int main() {
    
    // Run test model
    // DNNExample::xorModel();

    #ifdef _OPENMP
    cout << "OpenMP is supported" << endl;
    #endif

    typedef Matrix<double, Dynamic, Dynamic> MatrixD;

    /*
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
    */

    MaxPoolLayer<double> mp(3, 4, 4, 2, 2, 0);

    MatrixD input(4, 4);
    input << 1, 2, 3, 4,
             5, 6, 7, 8,
             9, 10, 11, 12,
             13, 14, 15, 16;

    vector<MatrixD> x = {input, input, input};



    vector<MatrixD> output = mp.forward(x);

    cout << "Input: " << endl << input << endl;
    cout << "Output: " << endl << output[0] << endl;

    MatrixD fake_res(2, 2);
    fake_res << 1, 2,
                   3, 4;

    vector<MatrixD> output_grad = {fake_res, fake_res, fake_res};

    vector<MatrixD> input_grad = mp.backward(output_grad, 0.01);

    cout << "Output gradient: " << endl << output_grad[0] << endl;
    cout << "Input gradient: " << endl << input_grad[0] << endl;
    
    return 0;
} 

