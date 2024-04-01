#include "DeepNeuralNetwork.hpp"
#include "XorModel.cpp"
// #include "ConvolutionalLayer.hpp"
#include "MaxPoolLayer.hpp"
using std::cout, std::vector, std::unique_ptr;
using Eigen::Matrix, Eigen::Dynamic, Eigen::MatrixXd;;

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

    MaxPoolLayer<double> mp(5, 10000, 10000, 2, 2, 0);

    MatrixD input = MatrixD::Random(10000, 10000);

    vector<MatrixD> x = {input, input, input, input, input};

    vector<MatrixD> output = mp.forward(x);

    //cout << "Input: " << endl << input << endl;
    //cout << "Output: " << endl << output[0] << endl;

    MatrixD fake_res = MatrixD::Random(5000, 5000);

    vector<MatrixD> output_grad = {fake_res, fake_res, fake_res, fake_res, fake_res};

    vector<MatrixD> input_grad = mp.backward(output_grad, 0.01);
    //cout << "Output gradient: " << endl << output_grad[0] << endl;
    //cout << "Input gradient: " << endl << input_grad[0] << endl;
    
    return 0;
} 

