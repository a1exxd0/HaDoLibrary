#include <HaDo/DeepNeuralNetwork>
#include "XorModel.cpp"
#include <HaDo/ConvolutionalNeuralNetwork>
using std::cout, std::vector, std::unique_ptr;
using Eigen::Matrix, Eigen::Dynamic, Eigen::MatrixXd;
using MatrixD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

using namespace hado;

int main() {
    
    

    #ifdef _OPENMP
    cout << "OpenMP is supported" << endl;
    #endif

    // Run test model
    DNNExample::xorModel();

    // typedef Matrix<double, Dynamic, Dynamic> MatrixD;

    // // Serialiser

    // MatrixD matrix = MatrixD::Random(5, 5);
    // vector<MatrixD> matrices = {matrix, matrix, matrix};
    // TensorSerializer<double>::write_to_file_test("src/models/matrix.json", matrices);
    // vector<MatrixD> matrix2;
    // TensorSerializer<double>::read_from_file_test("src/models/matrix.json", matrix2);
    // cout << "Matrix: " << endl << matrix2[0] << endl;



    ConvolutionalLayer<double, sigmoid<double>, sigmoid_prime<double>> mp(3,2,6,6,2,1,0);

    MatrixD input = MatrixD::Random(6, 6);

    vector<MatrixD> x = {input, input, input};

    vector<MatrixD> output = mp.forward(x);

    cout << "Input: " << endl << input << endl;
    cout << "Output: " << endl << output[0] << endl;

    MatrixD fake_res = MatrixD::Random(5, 5);

    vector<MatrixD> output_grad = {fake_res, fake_res, fake_res};

    vector<MatrixD> input_grad = mp.backward(output_grad, 0.01);
    cout << "Output gradient: " << endl << output_grad[0] << endl;
    cout << "Input gradient: " << endl << input_grad[0] << endl;
    
    return 0;
} 

