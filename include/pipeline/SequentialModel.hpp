#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>
#include <utility>
#include <Eigen/Dense>
#include "Layer.hpp"
#include "EndLayer.hpp"
#include "LayerVector.hpp"
#include "Pipeline.hpp"
#include <memory>

using std::vector;
using std::pair;
using std::unique_ptr;
using Eigen::Matrix, Eigen::Dynamic;

/**
 * @brief Full model structure. Does NOT enforce data dimension correctness.
 * 
 * @tparam T scalar type (float, double, long double)
*/
template<typename T=float>
class SequentialModel {
private:

    // Convenience typedef
    typedef Matrix<T, Dynamic, Dynamic> MatrixD;

    // Assert that T is either float, double, or long double at compiler time
    #pragma GCC diagnostic ignored "-Wparentheses"
    static_assert(
        std::is_same<T, float>::value 
        || std::is_same<T, double>::value
        || std::is_same<T, long double>::value
    );

    // Data holders for automatic training and testing
    vector<vector<MatrixD>> training_data;
    vector<vector<MatrixD>> training_results;
    vector<vector<MatrixD>> test_data;
    vector<vector<MatrixD>> test_results;

public:

    /**
     * @brief Construct a new Model object
     * 
     * @param pipeline Pipeline object to use
    */
    SequentialModel(Pipeline<T>& pipeline){
        this->pipeline = pipeline.clone();
    }

    // Copy constructor
    SequentialModel(const SequentialModel& m){
        pipeline = m.pipeline->clone();
        training_data = m.training_data;
        training_results = m.training_results;
        test_data = m.test_data;
        test_results = m.test_results;
    }

    // Clone returning unique ptr
    unique_ptr<Layer<T>> clone() const {
        return std::make_unique<SequentialModel<T>>(*this);
    }

    // Destructor
    ~SequentialModel(){}

    // Pipeline object public for direct access
    unique_ptr<Pipeline<T>> pipeline;

    /**
     * @brief Add single item of training data to the model with result.
     * Tensor dimensions must matcch input into model, but is not enforced
     * until runtime,
     * 
     * @param data input tensor
     * @param res result tensor
    */
    void add_training_data(vector<MatrixD> data, vector<MatrixD> res){
        training_data.push_back(data);
        training_results.push_back(res);
    }

    /**
     * @brief Add single item of test data to the model with result.
     * Tensor dimensions must matcch input into model, but is not enforced
     * until runtime,
     * 
     * @param data input tensor
     * @param res result tensor
    */
    void add_test_data(vector<MatrixD> data, vector<MatrixD> res){
        if (data.size() != res.size()){
            cout << "Data and result vectors must be of same size." << endl;
            exit(1);
        }
        test_data.push_back(data);
        test_results.push_back(res);
    }

    /**
     * @brief Set all training data at once. Dimension checks not ran until runtime.
     * 
     * @param data input tensor
     * @param res result tensor
    */
    void set_training_data(vector<vector<MatrixD>> data, vector<vector<MatrixD>> res){
        if (data.size() != res.size()){
            cout << "Data and result vectors must be of same size." << endl;
            exit(1);
        }
        training_data = data;
        training_results = res;
    }

    /**
     * @brief Set all test data at once. Dimension checks not ran until runtime.
     * 
     * @param data input tensor
     * @param res result tensor
    */
    void set_test_data(vector<vector<MatrixD>> data, vector<vector<MatrixD>> res){
        test_data = data;
        test_results = res;
    }

    /**
     * @brief Run a number of epochs with a given learning rate.
     * 
     * @param epochs number of epochs to run over data
     * @param learning_rate learning rate for model
     * @param to_print number of epochs to print error for. Default 100.
    */
    void run_epochs(const int epochs, const T learning_rate, int to_print=100){
        if(to_print < 0) to_print = 100;
        cout << "Running " << epochs << " epochs with learning rate ";
        cout << learning_rate << ":" << endl << endl;

        // Set up print factor and data size
        const int print_factor = (to_print == 0) ? 2147483647 : epochs / to_print;
        const int n = training_data.size();

        // Run epochs
        for(int epoch = 1; epoch < epochs+1; epoch++){
            // Sum error for each piece of input data
            T cumulative_error = 0;
            for(int i = 0; i < n; i++){
                // Train model
                cumulative_error += pipeline->trainPipeline(
                    training_data[i],
                    training_results[i],
                    learning_rate
                );
            }

            // Print error average across singular epoch
            if (epoch % print_factor == 0){
                cout << "Epoch " << epoch << " - Error: " << (cumulative_error / n) << endl;
            }
        }
    }

    /**
     * @brief Run tests on the model with test data.
     * 
     * @param to_print number of tests to print error for. Defaults to all.
    */
    void run_tests(int to_print=-1){
        if (to_print < 0) to_print = test_data.size();
        cout << "\n\nRunning tests...\n\n";

        // Set up print factor and data size
        const int n = test_data.size();
        const int print_factor = (to_print == 0) ? 2147483647 : n / to_print;

        // Run test data set
        T cumulative_error = 0;

        for(int i = 0; i < n; i++){
            // Calculate individual errors and add to total
            auto individual_error = pipeline->testPipeline(
                test_data[i],
                test_results[i]
            );
            cumulative_error += individual_error.first;
            if (i % print_factor == 0){
                cout << "Item " << i << " - Error: " << individual_error.first << endl;
            }
        }
        // Return average error
        cout << "Average error: " << (cumulative_error / n) << endl;
    }

};

#endif // MODEL_HPP