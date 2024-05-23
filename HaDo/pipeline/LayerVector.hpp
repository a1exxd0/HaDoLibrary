#ifndef LAYER_VECTOR_HPP
#define LAYER_VECTOR_HPP

#include <Eigen/Dense>
#include <vector>
#include "Layer.hpp"
#include "EndLayer.hpp"
#include <memory>
#include <type_traits>

using Eigen::Matrix;
using std::vector;
using std::unique_ptr;

namespace hado {


/**
 * @brief Layer vector class.
 * 
 * @details Layer vector class that stores layers in a vector.
 * 
 * @tparam T Data type (float for speed, double accuracy) (optional)
*/
template <typename T=float>
class LayerVector {
private:

    // Convenience typedef
    typedef Matrix<T, Dynamic, Dynamic> MatrixD;

    // Abstract away depth/dimension parameters inside container
    int entry_depth;
    int entry_rows;
    int entry_cols;
    int final_depth;
    int final_rows;
    int final_cols;

    // List of pointers to layers, using type erasure
    vector<unique_ptr<Layer<T>>> layers;

    /**
     * @brief Push a layer onto the empty container
     * 
     * @tparam LayerType type of layer (not base), i.e. DenseLayer
     * 
     * @param layer1 Layer of LayerType to be added
    */
    template<typename LayerType>
    void pushEmpty(LayerType layer1) {
        layers.push_back(std::make_unique<LayerType>(layer1));
        entry_depth = layers[0]->getInputDepth();
        entry_rows = layers[0]->getInputRows();
        entry_cols = layers[0]->getInputCols();
        final_depth = layers[0]->getOutputDepth();
        final_rows = layers[0]->getOutputRows();
        final_cols = layers[0]->getOutputCols();
    }
    
public:

    // Getters
    int getFinalDepth() { return final_depth; }
    int getFinalRows() { return final_rows; }
    int getFinalCols() { return final_cols; }

    // Default constructor
    LayerVector() {};

    // Clone returning unique pointer of class
    unique_ptr<LayerVector<T>> clone() const {
        unique_ptr<LayerVector<T>> res(new LayerVector<T>());
        for (auto& layer : layers) {
            res->layers.push_back(layer->clone());
            res->entry_depth = entry_depth;
            res->entry_rows = entry_rows;
            res->entry_cols = entry_cols;
            res->final_depth = final_depth;
            res->final_rows = final_rows;
            res->final_cols = final_cols;
        }
        return res;
    }

    // Destructor
    ~LayerVector() {};

    /**
     * @brief Push a layer onto the end of the vector
     * 
     * @tparam LayerType type of layer (not base), i.e. DenseLayer
     * 
     * @param layer Layer of LayerType to be added
    */
    template<typename LayerType>
    void pushLayer(LayerType layer){
        static_assert(std::is_base_of<Layer<T>, LayerType>::value,
                  "LayerType must derive from Layer<T>");

        // Call empty adder if empty pipeline
        if (layers.size() == 0) {
            pushEmpty<LayerType>(layer);
            return;
        }
        if (final_depth != layer.getInputDepth() 
            || final_rows != layer.getInputRows() 
            || final_cols != layer.getInputCols()) {
                cout << "Output depth from last layer: " << final_depth 
                    << " must match input depth " << layer.getInputDepth() << endl;

                cout << "Output rows from last layer: " << final_rows 
                    << " must match input rows " << layer.getInputRows() << endl;

                cout << "Output cols from last layer: " << final_cols 
                    << " must match input cols " << layer.getInputCols() << endl;
                throw std::invalid_argument("Layer dimensions must match previous layer dimensions.");
            }

        // Allocate memory and push onto vector
        layers.push_back(std::make_unique<LayerType>(layer));

        // Update attributes
        final_depth = layers.back()->getOutputDepth();
        final_rows = layers.back()->getOutputRows();
        final_cols = layers.back()->getOutputCols();
    }

    /**
     * @brief Send an input through the model and get result at end of pipe.
     * Input dimensions must match container inout dimensions.
     * 
     * @param input vector<MatrixD> to send forward
    */
    vector<MatrixD> forward(vector<MatrixD> input){

        // Dimension check
        if (input.size() != (size_t) this->entry_depth 
            || input[0].rows() != this->entry_rows 
            || input[0].cols() != this->entry_cols) {
                cout << "Input tensor must have depth " << entry_depth 
                    << " but got depth " << input.size() << endl;

                cout << "Input tensor must have rows " << entry_rows 
                    << " but got rows " << input[0].rows() << endl;

                cout << "Input tensor must have cols " << entry_cols 
                    << " but got cols " << input[0].cols() << endl;
                throw std::invalid_argument("Input tensor has incorrect dimensions." );
        }

        // Send the input and propagate forwards to end of model
        for (auto& layer : layers) {
            input = layer->forward(input);
        }

        // Return the resultant vector
        return input;
    }
    
    /**
     * @brief Given the gradient of error in error function, backwards propagate
     * the gradient through the model to perform stochastic gradient descent.
     * 
     * @param output_gradient Gradient of error from result of error function
     * @param learning_rate Learning rate of model
    */
    vector<MatrixD> backward(vector<MatrixD> output_gradient, const T learning_rate){

        // Dimension check
        if (output_gradient.size() != (size_t) final_depth 
            || output_gradient[0].rows() != final_rows 
            || output_gradient[0].cols() != final_cols) {
                cout << "Output gradient tensor must have depth " << final_depth 
                    << " but got depth " << output_gradient.size() << endl;

                cout << "Output gradient tensor must have rows " << final_rows 
                    << " but got rows " << output_gradient[0].rows() << endl;

                cout << "Output gradient tensor must have cols " << final_cols 
                    << " but got cols " << output_gradient[0].cols() << endl;
                throw std::invalid_argument("Output gradient tensor has incorrect dimensions." );
            }

        // Backward propagate gradients
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            output_gradient = (*it)->backward(output_gradient, learning_rate);
        }

        // Return top gradient (usually meaningless)
        return output_gradient;
    }
};


}

#endif // LAYER_VECTOR_HPP
