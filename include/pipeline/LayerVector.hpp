#ifndef LAYER_VECTOR_HPP
#define LAYER_VECTOR_HPP

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "Layer.hpp"
#include "EndLayer.hpp"

using Eigen::Matrix;
using std::vector;
using std::unique_ptr;
using std::make_unique;

/**
 * 
*/
class Pipeline {
private:

    vector<unique_ptr<Layer>> layers;
    int entry_depth;
    int entry_rows;
    int entry_cols;
    int final_depth;
    int final_rows;
    int final_cols;
    
public:

    Pipeline(unique_ptr<Layer> init_layer) {
        layers.push_back(std::move(init_layer));
        entry_depth = layers[0]->getInputDepth();
        entry_rows = layers[0]->getInputRows();
        entry_cols = layers[0]->getInputCols();
        final_depth = layers[0]->getOutputDepth();
        final_rows = layers[0]->getOutputRows();
        final_cols = layers[0]->getOutputCols();
    }

    ~Pipeline() {}

    
};

#endif // LAYER_VECTOR_HPP