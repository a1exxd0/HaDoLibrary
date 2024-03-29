#ifndef PIPELINE_HPP
#define PIPELINE_HPP

#include <vector>
#include <utility>
#include <Eigen/Dense>
#include "Layer.hpp"
#include "EndLayer.hpp"
#include "LayerVector.hpp"
#include <memory>

using std::vector;
using std::pair;
using Eigen::Matrix, Eigen::Dynamic;


/**
 * @brief Pipeline class for automatic handling of network structure.
 * Single runs and does not handle epoch parameters.
 * 
 * @tparam T numeric type: float, double, long double
*/
template <typename T=float>
class Pipeline {
    
    // Convenience typedef
    typedef Matrix<T, Dynamic, Dynamic> MatrixD;

    // Assert that T is either float, double, or long double at compiler time
    #pragma GCC diagnostic ignored "-Wparentheses"
    static_assert(
        std::is_same<T, float>::value 
        || std::is_same<T, double>::value
        || std::is_same<T, long double>::value
    );

    // Vector for layer passing
    LayerVector<T>* layervector;

    // Error/ error gradient calculation layer
    EndLayer<T>* endlayer;

public:

    // Default constructor
    Pipeline() {
        layervector = new LayerVector<T>();
        endlayer = nullptr;
    }

    // Default destructor
    ~Pipeline() {
        delete layervector;
        delete endlayer;
    }

    /**
     * @brief Add a layer to the pipeline. Must match dimensions of previous
     * layer, and also cannot be used after endlayer is added.
     * 
     * @tparam LayerType type of layer to be added
     * @param layer LayerType layer to be added
    */
    template <typename LayerType>
    void pushLayer(LayerType layer) {

        // must push normal layers first
        if (endlayer != nullptr) {
            throw std::invalid_argument("End layer must be pushed last.");
        }

        layervector->template pushLayer<LayerType>(layer);
    }

    /**
     * @brief Add an error function to the model. Can override other error functions,
     * but must match dimensions of output from standard layers.
     * 
     * @tparam EndLayerType type of error function deriving EndLayer to add
     * @param end EndLayerType error function to add
    */
    template <typename EndLayerType>
    void pushEndLayer(EndLayerType end) {
        
        if (endlayer != nullptr) {
            delete endlayer;
        }

        // Must match dimensions
        if ((*layervector).getFinalDepth() != end.getDepth() 
            || (*layervector).getFinalRows() != end.getRows() 
            || (*layervector).getFinalCols() != end.getCols()) {
                cout << "Output depth from last layer: " << (*layervector).getFinalDepth() 
                    << " must match input depth " << end.getDepth()  << endl;

                cout << "Output rows from last layer: " << (*layervector).getFinalRows() 
                    << " must match input rows " << end.getRows() << endl;

                cout << "Output cols from last layer: " << (*layervector).getFinalCols() 
                    << " must match input cols " << end.getCols() << endl;
                throw std::invalid_argument("Layer dimensions must match previous layer dimensions.");
            }

        // Allocate memory and push
        EndLayerType* temp = new EndLayerType(end);
        this->endlayer = temp;
    }

    /**
     * @brief Train the network with a forward and backward propagation
     * 
     * @param input Input tensor into pipeline
     * @param true_res Expected result from pipeline
     * @param learning_rate Learning rate for model
     * @return Error of this forward propagation
    */
    T trainPipeline(vector<MatrixD>& input, vector<MatrixD>& true_res, T learning_rate) {

        // Send through network forward
        auto x = (*layervector).forward(input);

        // Calculate error
        T error = (*endlayer).forward(x, true_res);

        // Calculate derivative of error
        auto grad = (*endlayer).backward(x, true_res);

        // Backpropagate
        (*layervector).backward(grad, learning_rate);

        // Return error
        return error;
    }

    /**
     * @brief Runs an input through the model (forward only) with correct input and calculates error
     * 
     * @param input Input tensor into pipeline
     * @param true_res Expected result from pipeline
     * @return Error of forward propagation
    */
    pair<T, vector<MatrixD>> testPipeline(vector<MatrixD>& input, vector<MatrixD>& true_res) {

        // Send forward through network
        auto x = (*layervector).forward(input);

        // Calculate error
        T error = (*endlayer).forward(x, true_res);

        // Return error and result
        return std::make_pair(error, x);
    }

    /**
     * @brief Runs an input through the model (forward only)
     * 
     * @param input Input tensor into pipeline
     * @param true_res Expected result from pipeline
     * @return Error of forward propagation
    */
    vector<MatrixD> predictPipeline(vector<MatrixD>& input) {

        // Send forward through network and return result
        return (*layervector).forward(input);
    }
};

#endif // PIPELINE_HPP