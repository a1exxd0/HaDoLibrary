#include "DeepNeuralNetwork.hpp"
namespace DNNExample {

using Eigen::Matrix, Eigen::MatrixXf;

void xorModel() {

    // Instantiate pipeline and add layers
    Pipeline<float> pipeline;

    pipeline.pushLayer<DenseLayer<>>(
        DenseLayer<>(2, 3)
    );

    pipeline.pushLayer<ActivationLayer<f_tanh<>, f_tanh_prime<>>>(
        ActivationLayer<f_tanh<>, f_tanh_prime<>>(1, 3, 1)
    );

    pipeline.pushLayer<DenseLayer<>>(
        DenseLayer<>(3, 5)
    );

    pipeline.pushLayer<ActivationLayer<f_tanh<>, f_tanh_prime<>>>(
        ActivationLayer<f_tanh<>, f_tanh_prime<>>(1, 5, 1)
    );

    pipeline.pushLayer<DenseLayer<>>(
        DenseLayer<> (5, 3)
    );

    pipeline.pushLayer<ActivationLayer<f_tanh<>, f_tanh_prime<>>>(
        ActivationLayer<f_tanh<>, f_tanh_prime<>>(1,3,1)
    );

    pipeline.pushLayer<DenseLayer<>>(
        DenseLayer<>(3, 1)
    );

    pipeline.pushLayer<ActivationLayer<f_tanh<>, f_tanh_prime<>, float>>(
        ActivationLayer<f_tanh<>, f_tanh_prime<>>(1,1,1)
    );
    
    pipeline.pushEndLayer<MeanSquaredError<float>>(
        MeanSquaredError<>(1,1,1)
    );

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
    Model<> model(pipeline);

    MatrixXf input1(2, 1); input1 << 0, 0;
    MatrixXf input2(2, 1); input2 << 0, 1;
    MatrixXf input3(2, 1); input3 << 1, 0;
    MatrixXf input4(2, 1); input4 << 1, 1;
    
    MatrixXf true_res1(1, 1); true_res1 << 0;
    MatrixXf true_res2(1, 1); true_res2 << 1;

    model.add_training_data({input1}, {true_res1});
    model.add_training_data({input2}, {true_res2});
    model.add_training_data({input3}, {true_res2});
    model.add_training_data({input4}, {true_res1});

    model.add_test_data({input1}, {true_res1});
    model.add_test_data({input2}, {true_res2});
    model.add_test_data({input3}, {true_res2});
    model.add_test_data({input4}, {true_res1});

    // Train and test model
    model.run_epochs(1000, 0.01, 20);
    model.run_epochs(1000, 0.001, 20);
    model.run_tests();

}

}