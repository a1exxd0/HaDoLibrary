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
using Eigen::Matrix, Eigen::Dynamic;

/**
 * @brief Full model structure. Does NOT enforce data dimension correctness.
 * 
 * @tparam T scalar type (float, double, long double)
*/
template<typename T=float>
class Model {
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

    vector<vector<MatrixD>> training_data;
    vector<vector<MatrixD>> training_results;
    vector<vector<MatrixD>> test_data;
    vector<vector<MatrixD>> test_results;

public:

    Model(Pipeline<T>& pipeline){
        this->pipeline = pipeline;
    }

    ~Model(){}

    Pipeline<T> pipeline;

    void add_training_data(vector<MatrixD> data, vector<MatrixD> res){
        training_data.push_back(data);
        training_results.push_back(res);
    }

    void add_test_data(vector<MatrixD> data, vector<MatrixD> res){
        test_data.push_back(data);
        test_results.push_back(res);
    }

    void set_training_data(vector<vector<MatrixD>> data, vector<vector<MatrixD>> res){
        training_data = data;
        training_results = res;
    }

    void set_test_data(vector<vector<MatrixD>> data, vector<vector<MatrixD>> res){
        test_data = data;
        test_results = res;
    }

    void run_epochs(int epochs, T learning_rate, int to_print=100){
        if(to_print <= 0) to_print = 100;
        cout << "Running " << epochs << " epochs with learning rate ";
        cout << learning_rate << ":" << endl << endl;

        const int print_factor = epochs / to_print;
        const int n = training_data.size();

        for(int epoch = 1; epoch < epochs+1; epoch++){
            T cumulative_error = 0;
            for(int i = 0; i < n; i++){
                cumulative_error += pipeline.trainPipeline(
                    training_data[i],
                    training_results[i],
                    learning_rate
                );
            }
            if (epoch % print_factor == 0){
                cout << "Epoch " << epoch << " - Error: " << (cumulative_error / n) << endl;
            }
        }
    }

    void run_tests(int to_print=-1){
        if (to_print <= 0) to_print = test_data.size();
        cout << "Running tests...\n\n";
        const int n = test_data.size();
        const int print_factor = n / to_print;

        T cumulative_error = 0;
        for(int i = 0; i < n; i++){
            auto individual_error = pipeline.testPipeline(
                test_data[i],
                test_results[i]
            );
            cumulative_error += individual_error.first;
            if (i % print_factor == 0){
                cout << "Item " << i << " - Error: " << individual_error.first << endl;
            }
        }
        cout << "Average error: " << (cumulative_error / n) << endl;
    }

};

#endif // MODEL_HPP