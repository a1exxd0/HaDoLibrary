#include <gtest/gtest.h>
#include <HaDo/base/ActivationFunctions.hpp>

using namespace hado;

constexpr double TOLERANCE=1e-6;
constexpr double LOOSE_TOLERANCE=1;

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

// ---------- SIGMOID ---------- //

TEST(SIGMOID, SIGMOID_ZERO) {
    float expected_f = 0.5;
    double expected_d = 0.5;

    sigmoid<float> f;
    sigmoid<double> d;
    auto result_f = f(0.0);
    auto result_d = d(0.0);

    EXPECT_FLOAT_EQ(result_f, expected_f);
    EXPECT_NEAR(result_d, expected_d, TOLERANCE);   
}

TEST(SIGMOID, SIGMOID_POSITIVE) {
    float expected_f = 0.6681877721682206;
    double expected_d = 0.6681877721682206;

    sigmoid<float> f;
    sigmoid<double> d;
    auto result_f = f(0.7);
    auto result_d = d(0.7);

    EXPECT_FLOAT_EQ(result_f, expected_f);
    EXPECT_NEAR(result_d, expected_d, TOLERANCE);   
}

TEST(SIGMOID, SIGMOID_NEGATIVE) {
    float expected_f = 0.40131233988751425;
    double expected_d = 0.40131233988751425;

    sigmoid<float> f;
    sigmoid<double> d;
    auto result_f = f(-0.4);
    auto result_d = d(-0.4);

    EXPECT_FLOAT_EQ(result_f, expected_f);
    EXPECT_NEAR(result_d, expected_d, TOLERANCE);     
}

TEST(SIGMOID, SIGMOIDP_ZERO) {
    float expected_f = 0.25;
    double expected_d = 0.25;

    sigmoid_prime<float> f;
    sigmoid_prime<double> d;
    auto result_f = f(0.0);
    auto result_d = d(0.0);

    EXPECT_FLOAT_EQ(result_f, expected_f);
    EXPECT_NEAR(result_d, expected_d, TOLERANCE); 
}

TEST(SIGMOID, SIGMOIDP_POSITIVE) {
    float expected_f = 0.22171287329309072;
    double expected_d = 0.22171287329309072;

    sigmoid_prime<float> f;
    sigmoid_prime<double> d;
    auto result_f = f(0.7);
    auto result_d = d(0.7);

    EXPECT_FLOAT_EQ(result_f, expected_f);
    EXPECT_NEAR(result_d, expected_d, TOLERANCE);   
}

TEST(SIGMOID, SIGMOIDP_NEGATIVE) {
    float expected_f = 0.24026074574152248;
    double expected_d = 0.24026074574152248;

    sigmoid_prime<float> f;
    sigmoid_prime<double> d;
    auto result_f = f(-0.4);
    auto result_d = d(-0.4);

    EXPECT_FLOAT_EQ(result_f, expected_f);
    EXPECT_NEAR(result_d, expected_d, TOLERANCE);    
}

// ---------- TANH ---------- //

TEST(TANH, TANH_ZERO) {
    float expected_f = 0.0;
    double expected_d = 0.0;

    f_tanh<float> f;
    f_tanh<double> d;
    auto result_f = f(0.0);
    auto result_d = d(0.0);

    EXPECT_FLOAT_EQ(result_f, expected_f);
    EXPECT_NEAR(result_d, expected_d, TOLERANCE);   
}

TEST(TANH, TANH_POSITIVE) {
    float expected_f = 0.761594155956;
    double expected_d = 0.761594155956;

    f_tanh<float> f;
    f_tanh<double> d;
    auto result_f = f(1.0);
    auto result_d = d(1.0);

    EXPECT_FLOAT_EQ(result_f, expected_f);
    EXPECT_NEAR(result_d, expected_d, TOLERANCE);   
}

TEST(TANH, TANH_NEGATIVE) {
    float expected_f = -0.964027580076;
    double expected_d = -0.964027580076;

    f_tanh<float> f;
    f_tanh<double> d;
    auto result_f = f(-2.0);
    auto result_d = d(-2.0);

    EXPECT_FLOAT_EQ(result_f, expected_f);
    EXPECT_NEAR(result_d, expected_d, TOLERANCE);     
}

TEST(TANH, TANHP_ZERO) {
    float expected_f = 1;
    double expected_d = 1;

    f_tanh_prime<float> f;
    f_tanh_prime<double> d;
    auto result_f = f(0.0);
    auto result_d = d(0.0);

    EXPECT_FLOAT_EQ(result_f, expected_f);
    EXPECT_NEAR(result_d, expected_d, TOLERANCE);   
}

TEST(TANH, TANHP_POSITIVE) {
    float expected_f = 0.257433196703;
    double expected_d = 0.257433196703;

    f_tanh_prime<float> f;
    f_tanh_prime<double> d;
    auto result_f = f(1.3);
    auto result_d = d(1.3);

    EXPECT_FLOAT_EQ(result_f, expected_f);
    EXPECT_NEAR(result_d, expected_d, TOLERANCE);    
}

TEST(TANH, TANHP_NEGATIVE) {
    float expected_f = 0.00134095068303;
    double expected_d = 0.00134095068303;

    f_tanh_prime<float> f;
    f_tanh_prime<double> d;
    auto result_f = f(-4.0);
    auto result_d = d(-4.0);

    EXPECT_NEAR(result_f, expected_f, TOLERANCE);
    EXPECT_NEAR(result_d, expected_d, TOLERANCE);    
}

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}