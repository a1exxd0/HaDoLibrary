#include <gtest/gtest.h>
#include <DeepNeuralNetwork>
using Eigen::Matrix, Eigen::MatrixXf;

using namespace hado;

TEST(XOR_MODEL, XOR_MODEL_RUN) {
    Pipeline<float> pipeline;

    pipeline.pushLayer(DenseLayer<>(2, 3));

    pipeline.pushLayer(ActivationLayer<f_tanh<>, f_tanh_prime<>>(1, 3, 1));

    pipeline.pushLayer(DenseLayer<>(3, 5));


    // You can optionally specify template for push if it isn't automatic
    pipeline.pushLayer(ActivationLayer<f_tanh<>, f_tanh_prime<>>(1, 5, 1));

    // You can also type deduce the DenseLayer template
    pipeline.pushLayer(DenseLayer(5, 3));


    pipeline.pushLayer(ActivationLayer<f_tanh<>, f_tanh_prime<>>(1,3,1));

    pipeline.pushLayer(DenseLayer(3, 1));

    pipeline.pushLayer(ActivationLayer<f_tanh<>, f_tanh_prime<>>(1,1,1));
    
    pipeline.pushEndLayer(MeanSquaredError<>(1,1,1));

    /**
     * SUMMARY OF LAYERS:
     * 
     * 1. DenseLayer: 2 input nodes, 3 output nodes
     * 2. ActivationLayer: tanh activation function
     * 3. DenseLayer: 3 input nodes, 5 output nodes
     * 4. ActivationLayer: tanh activation function
     * 5. DenseLayer: 5 input nodes, 3 output nodes
     * 6. ActivationLayer: tanh activation function
     * 7. DenseLayer: 3 input nodes, 1 output node
     * 8. ActivationLayer: tanh activation function
     * 9. MeanSquaredError: Loss function
    */

    // Instantiate model and add training and test data
    // Will predict the XOR of 2 inputs
    SequentialModel<float> model(pipeline);

    MatrixXf input1(2, 1); input1 << 0, 0;
    MatrixXf input2(2, 1); input2 << 0, 1;
    MatrixXf input3(2, 1); input3 << 1, 0;
    MatrixXf input4(2, 1); input4 << 1, 1;
    
    MatrixXf true_res1(1, 1); true_res1 << 0;
    MatrixXf true_res2(1, 1); true_res2 << 1;

    model.add_training_data({input1}, {true_res1}); // 0 XOR 0 = 0
    model.add_training_data({input2}, {true_res2}); // 0 XOR 1 = 1
    model.add_training_data({input3}, {true_res2}); // 1 XOR 0 = 1
    model.add_training_data({input4}, {true_res1}); // 1 XOR 1 = 0

    model.add_test_data({input1}, {true_res1});
    model.add_test_data({input2}, {true_res2});
    model.add_test_data({input3}, {true_res2});
    model.add_test_data({input4}, {true_res1});

    // Train and test model
    model.run_epochs(1000, 0.01, 10);
    //model.run_epochs(1000, 0.001, 0);
    auto err = model.run_tests();

    ASSERT_TRUE(err < 0.1);
}

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}