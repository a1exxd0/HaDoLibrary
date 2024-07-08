#include <HaDo/ConvolutionalNeuralNetwork>
#include <gtest/gtest.h>

using Eigen::Matrix, Eigen::MatrixXf;
using namespace hado;

TEST(BASIC_CONVOLUTIONAL_SETUP, BASIC_CONVOLUTIONAL_LAYER_SETUP_STRIDE_1) {
    auto layer = ConvolutionalLayer<double, f_tanh<double>, f_tanh_prime<double>>(
        2,1,4,4,1,1,0);

    // cout << "\n";
    // cout << "output rows: " << layer.getOutputRows() << endl;
    // cout << "output cols: " << layer.getOutputCols() << endl;
    // cout << "output depth: " << layer.getOutputDepth() << endl;

    EXPECT_EQ(layer.getOutputRows(), 4);
    EXPECT_EQ(layer.getOutputCols(), 4);
    EXPECT_EQ(layer.getOutputDepth(), 1);
}

TEST(BASIC_CONVOLUTIONAL_SETUP, PIPELINE_CONVOLUTIONAL_FLATTEN) {
    Pipeline<float> pipeline;

    pipeline.pushLayer(
        ConvolutionalLayer<float, f_tanh<>, f_tanh_prime<>>(
            1,1,4,4,2,1, 0)
    );

    pipeline.pushLayer(
        FlatteningLayer<>(1, 3, 3)
    );

    pipeline.pushEndLayer(
        MeanSquaredError<>(1, 1, 9)
    );

    SequentialModel<> model {pipeline};

    MatrixXf input1 = MatrixXf::Random(4, 4);
    MatrixXf expected1 = MatrixXf::Random(1, 9);

    model.add_training_data({input1}, {expected1});

    model.run_epochs(1, 0.01, 1);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}