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

template <typename L1, typename... Ts>
class LayerVector {
private:

    int entry_depth;
    int entry_rows;
    int entry_cols;
    int final_depth;
    int final_rows;
    int final_cols;
    
public:

    

    ~LayerVector() {}

    
};

#endif // LAYER_VECTOR_HPP