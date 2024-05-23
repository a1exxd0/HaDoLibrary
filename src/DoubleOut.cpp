#include <HaDo/DeepNeuralNetwork>
namespace DNNExample {

using Eigen::Matrix, Eigen::MatrixXf;
using namespace hado;

// TEST PURPOSES
void doubleOutModel() {

    // Instantiate pipeline and add layers
    Pipeline<float> pipeline;

    pipeline.pushLayer(
        DenseLayer(4, 6)
    );

    pipeline.pushLayer(
        ActivationLayer<f_tanh<>, f_tanh_prime<>>(1, 6, 1)
    );

    pipeline.pushLayer<DenseLayer<>>(
        DenseLayer(6, 4)
    );

    pipeline.pushLayer(
        ActivationLayer<f_tanh<>, f_tanh_prime<>>(1,4,1)
    );

    pipeline.pushLayer(
        DenseLayer(4, 2)
    );

    pipeline.pushLayer(
        SoftmaxLayer(2)
    );
    
    pipeline.pushEndLayer(
        CrossEntropyLoss(2)
    );

    // Instantiate model and add training and test data
    SequentialModel<float> model(pipeline);

    MatrixXf input1(4, 1); input1 << 1, 0, 0, 0;
    MatrixXf input2(4, 1); input2 << 0, 1, 0, 1;
    MatrixXf input3(4, 1); input3 << 0, 0, 1, 0;
    MatrixXf input4(4, 1); input4 << 0, 0, 0, 1;
    
    MatrixXf true_res1(2, 1); true_res1 << 0, 1;
    MatrixXf true_res2(2, 1); true_res2 << 1, 0;

    model.add_training_data({input1}, {true_res2});
    model.add_training_data({input2}, {true_res1});
    model.add_training_data({input3}, {true_res2});
    model.add_training_data({input4}, {true_res1});

    model.add_test_data({input1}, {true_res2});
    model.add_test_data({input2}, {true_res1});
    model.add_test_data({input3}, {true_res2});
    model.add_test_data({input4}, {true_res1});

    // Train and test model
    model.run_epochs(50000, 0.01, 10);
    //model.run_epochs(1000, 0.001, 0);
    //model.run_tests();

}

}