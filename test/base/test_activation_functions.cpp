#include <gtest/gtest.h>
#include <HaDo/base/ActivationFunctions.hpp>

using namespace hado;

// ---------- RELU ---------- //

TEST(RELU, RELU_ZERO) {
    float expected_f = 0.0;
    double expected_d = 0.0;

    relu<float> f;
    relu<double> d;
    auto result_f = f(0.0);
    auto result_d = d(0.0);

    EXPECT_EQ(result_f, expected_f);
    EXPECT_EQ(result_d, expected_d);    
}

TEST(RELU, RELU_POSITIVE) {
    float expected_f = 100.0;
    double expected_d = 100.0;

    relu<float> f;
    relu<double> d;
    auto result_f = f(100.0);
    auto result_d = d(100.0);

    EXPECT_EQ(result_f, expected_f);
    EXPECT_EQ(result_d, expected_d);    
}

TEST(RELU, RELU_NEGATIVE) {
    float expected_f = 0.0;
    double expected_d = 0.0;

    relu<float> f;
    relu<double> d;
    auto result_f = f(-100.0);
    auto result_d = d(-100.0);

    EXPECT_EQ(result_f, expected_f);
    EXPECT_EQ(result_d, expected_d);    
}

TEST(RELU, RELUP_ZERO) {
    float expected_f = 0.0;
    double expected_d = 0.0;

    relu_prime<float> f;
    relu_prime<double> d;
    auto result_f = f(0.0);
    auto result_d = d(0.0);

    EXPECT_EQ(result_f, expected_f);
    EXPECT_EQ(result_d, expected_d);    
}

TEST(RELU, RELUP_POSITIVE) {
    float expected_f = 1.0;
    double expected_d = 1.0;

    relu_prime<float> f;
    relu_prime<double> d;
    auto result_f = f(100.0);
    auto result_d = d(100.0);

    EXPECT_EQ(result_f, expected_f);
    EXPECT_EQ(result_d, expected_d);    
}

TEST(RELU, RELUP_NEGATIVE) {
    float expected_f = 0.0;
    double expected_d = 0.0;

    relu_prime<float> f;
    relu_prime<double> d;
    auto result_f = f(-100.0);
    auto result_d = d(-100.0);

    EXPECT_EQ(result_f, expected_f);
    EXPECT_EQ(result_d, expected_d);    
}

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}