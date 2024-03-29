#ifndef LAYER_VECTOR_HPP
#define LAYER_VECTOR_HPP

#include <Eigen/Dense>
#include <vector>
#include "Layer.hpp"
#include "EndLayer.hpp"

using Eigen::Matrix;
using std::vector;


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
    // Can't use smart pointers here so must be careful
    vector<Layer<T>*> layers;

    /**
     * @brief Push a layer onto the empty container
     * 
     * @tparam LayerType type of layer (not base), i.e. DenseLayer
     * 
     * @param layer1 Layer of LayerType to be added
    */
    template<typename LayerType>
    void pushEmpty(LayerType layer1) {
        LayerType* temp = new LayerType(layer1);
        layers.push_back(temp);
        entry_depth = temp->getInputDepth();
        entry_rows = temp->getInputRows();
        entry_cols = temp->getInputCols();
        final_depth = temp->getOutputDepth();
        final_rows = temp->getOutputRows();
        final_cols = temp->getOutputCols();
    }
    
public:

    // Getters
    int getFinalDepth() { return final_depth; }
    int getFinalRows() { return final_rows; }
    int getFinalCols() { return final_cols; }

    // Default constructor
    LayerVector() {};

    // Destructor
    ~LayerVector() {
        for (auto layer : layers) {
            delete layer;
        }
    }

    /**
     * @brief Push a layer onto the end of the vector
     * 
     * @tparam LayerType type of layer (not base), i.e. DenseLayer
     * 
     * @param layer Layer of LayerType to be added
    */
    template<typename LayerType>
    void pushLayer(LayerType layer){

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
        LayerType* temp = new LayerType(layer);
        layers.push_back(temp);

        // Update attributes
        final_depth = layer.getOutputDepth();
        final_rows = layer.getOutputRows();
        final_cols = layer.getOutputCols();
    }

    /**
     * @brief Safe removal method to get rid of a layer at the end
     * of the data structure
    */
    void popLayer(){
        // Free a layer allocated in pushLayer
        delete (layers.back());

        // Update attributes
        layers.pop_back();
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
        if (input.size() != (size_t) entry_depth 
            || input[0].rows() != entry_rows 
            || input[0].cols() != entry_cols) {
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
    vector<MatrixD> backward(vector<MatrixD> output_gradient, T learning_rate){

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

#endif // LAYER_VECTOR_HPP