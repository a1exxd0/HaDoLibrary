#include <gtest/gtest.h>
#include <HaDo/ConvolutionalNeuralNetwork>

using namespace hado;
using MatrixD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

using ConvolutionalLayerReLU = ConvolutionalLayer<double, relu<double>, relu_prime<double>>;
using ConvolutionalLayerSigmoid = ConvolutionalLayer<double, sigmoid<double>, sigmoid_prime<double>>;
using ConvolutionalLayerTanh = ConvolutionalLayer<double, f_tanh<double>, f_tanh_prime<double>>;

TEST(CONSTRUCTOR, Check_Negative_Zero_Sizes) {
    EXPECT_DEATH({
        ConvolutionalLayerReLU layer(1, 0, 3, 3, 3, 1, 0);
    }, "");

    EXPECT_DEATH({
        ConvolutionalLayerReLU layer(1, 10, -1, 3, 3, 1, 1);
    }, "");

    EXPECT_ANY_THROW({
        ConvolutionalLayerReLU layer(-1, 10, 3, 3, 3, 1, 1);
    });
}

TEST(CONSTRUCTOR, Standard_Constructor) {
    const ConvolutionalLayerReLU layer(2, 3, 5, 5, 3, 1, 1);
    ASSERT_EQ(layer.getInputDepth(), 2);
    ASSERT_EQ(layer.getInputCols(), 5);
    ASSERT_EQ(layer.getInputRows(), 5);
    ASSERT_EQ(layer.getOutputDepth(), 3);
    ASSERT_EQ(layer.getOutputCols(), 5);
    ASSERT_EQ(layer.getOutputRows(), 5);
    ASSERT_EQ(layer.getKernelSize(), 3);
    ASSERT_EQ(layer.getStride(), 1);
    ASSERT_EQ(layer.getPadding(), 1);
}

TEST(CONSTRUCTOR, Copy_Constructor) {
    const ConvolutionalLayerReLU layer(3, 10, 10, 10, 3, 1, 1);

    const ConvolutionalLayerReLU result(layer);

    ASSERT_EQ(result.getInputDepth(), 3);
    ASSERT_EQ(result.getInputCols(), 10);
    ASSERT_EQ(result.getInputRows(), 10);
    ASSERT_EQ(result.getOutputDepth(), 10);
    ASSERT_EQ(result.getOutputCols(), 10);
    ASSERT_EQ(result.getOutputRows(), 10);
    ASSERT_EQ(result.getKernelSize(), 3);
    ASSERT_EQ(result.getStride(), 1);
    ASSERT_EQ(result.getPadding(), 1);
    ASSERT_TRUE(&layer != &result);
}

TEST(CONSTRUCTOR, Clone_Constructor) {
    const ConvolutionalLayerReLU layer(3, 10, 10, 10, 3, 1, 1);

    const auto result = layer.clone();

    // Cast to ConvolutionalLayerReLU to access specific methods
    const ConvolutionalLayerReLU* cloned_layer = dynamic_cast<const ConvolutionalLayerReLU*>(result.get());

    ASSERT_NE(cloned_layer, nullptr);
    ASSERT_EQ(cloned_layer->getInputDepth(), 3);
    ASSERT_EQ(cloned_layer->getInputCols(), 10);
    ASSERT_EQ(cloned_layer->getInputRows(), 10);
    ASSERT_EQ(cloned_layer->getOutputDepth(), 10);
    ASSERT_EQ(cloned_layer->getOutputCols(), 10);
    ASSERT_EQ(cloned_layer->getOutputRows(), 10);
    ASSERT_EQ(cloned_layer->getKernelSize(), 3);
    ASSERT_EQ(cloned_layer->getStride(), 1);
    ASSERT_EQ(cloned_layer->getPadding(), 1);
    ASSERT_TRUE(&layer != cloned_layer);
}

TEST(FORWARD_BACKWARD, Forward_Backward_Pass_ReLU) {
    const int inputDepth = 2, outputDepth = 3, inputRows = 5, inputCols = 5, kernelSize = 3, stride = 1, padding = 1;
    ConvolutionalLayerReLU layer(inputDepth, outputDepth, inputRows, inputCols, kernelSize, stride, padding);

    // Mock input tensor
    vector<MatrixD> input_tensor(inputDepth, MatrixD::Random(inputRows, inputCols));

    // Forward pass
    auto output_tensor = layer.forward(input_tensor);

    ASSERT_EQ(output_tensor.size(), outputDepth);
    for (const auto& mat : output_tensor) {
        ASSERT_EQ(mat.rows(), layer.getOutputRows());
        ASSERT_EQ(mat.cols(), layer.getOutputCols());
    }

    // Mock output gradient tensor for backward pass (same dimensions as forward output)
    vector<MatrixD> output_gradient(outputDepth, MatrixD::Random(layer.getOutputRows(), layer.getOutputCols()));
    double learning_rate = 0.01;

    // Backward pass
    auto input_gradient = layer.backward(output_gradient, learning_rate);

    ASSERT_EQ(input_gradient.size(), inputDepth);
    for (const auto& mat : input_gradient) {
        ASSERT_EQ(mat.rows(), inputRows);
        ASSERT_EQ(mat.cols(), inputCols);
    }
}

TEST(FORWARD_BACKWARD, Forward_Backward_Pass_Sigmoid) {
    const int inputDepth = 2, outputDepth = 3, inputRows = 5, inputCols = 5, kernelSize = 3, stride = 1, padding = 1;
    ConvolutionalLayerSigmoid layer(inputDepth, outputDepth, inputRows, inputCols, kernelSize, stride, padding);

    // Mock input tensor
    vector<MatrixD> input_tensor(inputDepth, MatrixD::Random(inputRows, inputCols));

    // Forward pass
    auto output_tensor = layer.forward(input_tensor);

    ASSERT_EQ(output_tensor.size(), outputDepth);
    for (const auto& mat : output_tensor) {
        ASSERT_EQ(mat.rows(), layer.getOutputRows());
        ASSERT_EQ(mat.cols(), layer.getOutputCols());
    }

    // Mock output gradient tensor for backward pass (same dimensions as forward output)
    vector<MatrixD> output_gradient(outputDepth, MatrixD::Random(layer.getOutputRows(), layer.getOutputCols()));
    double learning_rate = 0.01;

    // Backward pass
    auto input_gradient = layer.backward(output_gradient, learning_rate);

    ASSERT_EQ(input_gradient.size(), inputDepth);
    for (const auto& mat : input_gradient) {
        ASSERT_EQ(mat.rows(), inputRows);
        ASSERT_EQ(mat.cols(), inputCols);
    }
}

TEST(FORWARD_BACKWARD, Forward_Backward_Pass_Tanh) {
    const int inputDepth = 2, outputDepth = 3, inputRows = 5, inputCols = 5, kernelSize = 3, stride = 1, padding = 1;
    ConvolutionalLayerTanh layer(inputDepth, outputDepth, inputRows, inputCols, kernelSize, stride, padding);

    // Mock input tensor
    vector<MatrixD> input_tensor(inputDepth, MatrixD::Random(inputRows, inputCols));

    // Forward pass
    auto output_tensor = layer.forward(input_tensor);

    ASSERT_EQ(output_tensor.size(), outputDepth);
    for (const auto& mat : output_tensor) {
        ASSERT_EQ(mat.rows(), layer.getOutputRows());
        ASSERT_EQ(mat.cols(), layer.getOutputCols());
    }

    // Mock output gradient tensor for backward pass (same dimensions as forward output)
    vector<MatrixD> output_gradient(outputDepth, MatrixD::Random(layer.getOutputRows(), layer.getOutputCols()));
    double learning_rate = 0.01;

    // Backward pass
    auto input_gradient = layer.backward(output_gradient, learning_rate);

    ASSERT_EQ(input_gradient.size(), inputDepth);
    for (const auto& mat : input_gradient) {
        ASSERT_EQ(mat.rows(), inputRows);
        ASSERT_EQ(mat.cols(), inputCols);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
