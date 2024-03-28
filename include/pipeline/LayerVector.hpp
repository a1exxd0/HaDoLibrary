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
template <typename T>
class LayerVector {
private:

    typedef Matrix<T, Dynamic, Dynamic> MatrixD;

    int entry_depth;
    int entry_rows;
    int entry_cols;
    int final_depth;
    int final_rows;
    int final_cols;
    vector<Layer<T>*> layers;

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

    // Default constructor
    LayerVector() {};

    // Destructor
    ~LayerVector() {
        for (auto layer : layers) {
            delete layer;
        }
    }

    template<typename LayerType>
    void pushLayer(LayerType layer){
        if (layers.size() == 0) {
            pushEmpty<LayerType>(layer);
            return;
        }
        if (final_depth != layer.getInputDepth() || final_rows != layer.getInputRows() || final_cols != layer.getInputCols()) {
            cout << "Output depth from last layer: " << final_depth << " must match input depth " << layer.getInputDepth() << endl;
            cout << "Output rows from last layer: " << final_rows << " must match input rows " << layer.getInputRows() << endl;
            cout << "Output cols from last layer: " << final_cols << " must match input cols " << layer.getInputCols() << endl;
            throw std::invalid_argument("Layer dimensions must match previous layer dimensions.");
        }
        LayerType* temp = new LayerType(layer);
        layers.push_back(temp);
        final_depth = layer.getOutputDepth();
        final_rows = layer.getOutputRows();
        final_cols = layer.getOutputCols();
    }

    void popLayer(){
        delete (layers.back());
        layers.pop_back();
        final_depth = layers.back()->getOutputDepth();
        final_rows = layers.back()->getOutputRows();
        final_cols = layers.back()->getOutputCols();
    }

    vector<MatrixD> forward(vector<MatrixD> input){
        if (input.size() != (size_t) entry_depth || input[0].rows() != entry_rows || input[0].cols() != entry_cols) {
            cout << "Input tensor must have depth " << entry_depth << " but got depth " << input.size() << endl;
            cout << "Input tensor must have rows " << entry_rows << " but got rows " << input[0].rows() << endl;
            cout << "Input tensor must have cols " << entry_cols << " but got cols " << input[0].cols() << endl;
            throw std::invalid_argument("Input tensor has incorrect dimensions." );
        }
        for (auto& layer : layers) {
            input = layer->forward(input);
        }
        return input;
    }
    
    vector<MatrixD> backward(vector<MatrixD> output_gradient, T learning_rate){
        if (output_gradient.size() != (size_t) final_depth || output_gradient[0].rows() != final_rows || output_gradient[0].cols() != final_cols) {
            cout << "Output gradient tensor must have depth " << final_depth << " but got depth " << output_gradient.size() << endl;
            cout << "Output gradient tensor must have rows " << final_rows << " but got rows " << output_gradient[0].rows() << endl;
            cout << "Output gradient tensor must have cols " << final_cols << " but got cols " << output_gradient[0].cols() << endl;
            throw std::invalid_argument("Output gradient tensor has incorrect dimensions." );
        }
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            output_gradient = (*it)->backward(output_gradient, learning_rate);
        }
        return output_gradient;
    }
};

#endif // LAYER_VECTOR_HPP