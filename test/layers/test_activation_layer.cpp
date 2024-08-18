#include <gtest/gtest.h>
#include <HaDo/DeepNeuralNetwork>

using namespace hado;
using MatrixD = Matrix<double, Dynamic, Dynamic>;
using Activation = ActivationLayer<relu<double>, relu_prime<double>, double>;

TEST(CONSTRUCTOR, Check_Negative_Zero_Sizes) {
    // This should check subclass attributes too
    
    EXPECT_DEATH({
        Activation x(1,0,3); 
    }, "");

    EXPECT_DEATH({
        Activation x(1,10,-1); 
    }, "");

    EXPECT_DEATH({
        Activation x(-1, 10, 3); 
    }, "");
}

TEST(CONSTRUCTOR, Standard_Constructor) {
    // This should check subclass attributes
    
    const Activation layer(2, 3, 4);
    ASSERT_EQ(layer.getInputDepth(), 2);
    ASSERT_EQ(layer.getInputCols(), 4);
    ASSERT_EQ(layer.getInputRows(), 3);
    ASSERT_EQ(layer.getOutputDepth(), 2);
    ASSERT_EQ(layer.getOutputCols(), 4);
    ASSERT_EQ(layer.getOutputRows(), 3);
}

TEST(CONSTRUCTOR, Copy_Constructor) {
    const Activation layer(3, 10, 10);

    const Activation result(layer);

    ASSERT_EQ(result.getInputDepth(), 3);
    ASSERT_EQ(result.getInputCols(), 10);
    ASSERT_EQ(result.getInputRows(), 10);
    ASSERT_EQ(result.getOutputDepth(), 3);
    ASSERT_EQ(result.getOutputCols(), 10);
    ASSERT_EQ(result.getOutputRows(), 10);
    ASSERT_TRUE(&layer != &result);
}

TEST(CONSTRUCTOR, Clone_Constructor) {
    const Activation layer(3, 10, 10);

    const auto result = layer.clone();

    ASSERT_EQ(result->getInputDepth(), 3);
    ASSERT_EQ(result->getInputCols(), 10);
    ASSERT_EQ(result->getInputRows(), 10);
    ASSERT_EQ(result->getOutputDepth(), 3);
    ASSERT_EQ(result->getOutputCols(), 10);
    ASSERT_EQ(result->getOutputRows(), 10);
    ASSERT_TRUE(&layer != &(*result));
}

TEST(FORWARD_BACKWARD, RELU_FORWARD) {
    Activation layer {1, 10, 10};

    vector<MatrixD> inp = {MatrixD::Random(10,10)};

    auto res = layer.forward(inp);

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            ASSERT_EQ(res[0](i,j), relu<double>{}(inp[0](i,j)));
        }
    }
}

TEST(FORWARD_BACKWARD, RELU_BACKWARD) {
    Activation layer {1, 10, 12};

    vector<MatrixD> inp = {MatrixD::Random(10,12)};
    vector<MatrixD> rev = {MatrixD::Random(10, 12)};

    auto in = layer.forward(inp);
    auto res = layer.backward(rev, 0);

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 12; ++j) {
            ASSERT_EQ(res[0](i,j),
                relu_prime<double>{}(inp[0](i,j)) * rev[0](i,j));
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}