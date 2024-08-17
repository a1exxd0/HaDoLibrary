#include <gtest/gtest.h>
#include <HaDo/DeepNeuralNetwork>

using namespace hado;
using Dense = DenseLayer<float>;

TEST(CONSTRUCTOR, CheckNegativeZeroSize) {
    EXPECT_DEATH({
        Dense x(0, 1);
    }, "");

    EXPECT_DEATH({
        Dense x(1, -1);
    }, "");

    EXPECT_DEATH({
        Dense x(-1, 4);
    }, "");
}

TEST(CONSTRUCTOR, StandardConstructor) {
    const Dense x {10, 15};

    ASSERT_EQ(x.getInputCols(), 1);
    ASSERT_EQ(x.getInputRows(), 10);
    ASSERT_EQ(x.getInputDepth(), 1);
    ASSERT_EQ(x.getOutputCols(), x.getInputCols());
    ASSERT_EQ(x.getOutputDepth(), x.getInputDepth());
    ASSERT_EQ(x.getOutputRows(), 15);
}

TEST(CONSTRUCTOR, CopyConstructor) {
    const Dense x {3, 2};
    const Dense y {x};

    ASSERT_EQ(x.getInputCols(), y.getInputCols());
    ASSERT_EQ(x.getInputRows(), y.getInputRows());
    ASSERT_EQ(x.getInputDepth(), y.getInputDepth());
    ASSERT_EQ(x.getOutputCols(), y.getOutputCols());
    ASSERT_EQ(x.getOutputDepth(), y.getOutputDepth());
    ASSERT_EQ(x.getOutputRows(), y.getOutputRows());
}

TEST(CONSTRUCTOR, CloneConstructor) {
    const Dense x {3, 2};
    const auto y_ptr = x.clone();
    const Layer<>& y = *y_ptr;

    ASSERT_EQ(x.getInputCols(), y.getInputCols());
    ASSERT_EQ(x.getInputRows(), y.getInputRows());
    ASSERT_EQ(x.getInputDepth(), y.getInputDepth());
    ASSERT_EQ(x.getOutputCols(), y.getOutputCols());
    ASSERT_EQ(x.getOutputDepth(), y.getOutputDepth());
    ASSERT_EQ(x.getOutputRows(), y.getOutputRows());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
