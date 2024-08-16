#include <gtest/gtest.h>
#include <HaDo/DeepNeuralNetwork>

using namespace hado;
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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}