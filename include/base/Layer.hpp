#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <iostream>

using Eigen::Matrix;
using std::vector;
using Eigen::Dynamic;
using std::cout;

#ifndef endl
    #define endl "\n"
#endif

/**
 * @brief Base layer class. T will only work for float, double,
 * or long double.
 * 
 * 
 * @tparam T Data type (float for speed, double accuracy) (optional)
*/
template<typename T=float>
class Layer{
private:

    int I, O, RI, CI, RO, CO;

    // Convenience typedef
    typedef Matrix<T, Dynamic, Dynamic> MatrixD;

    // Assert that T is either float, double, or long double at compiler time
    #pragma GCC diagnostic ignored "-Wparentheses"
    static_assert(
        std::is_same<T, float>::value 
        || std::is_same<T, double>::value
        || std::is_same<T, long double>::value
    );

protected:

    // Default constructor
    Layer(int I, int O, int RI, int CI, int RO, int CO) : inp(I), out(O) {
        this->I = I;
        this->O = O;
        this->RI = RI;
        this->CI = CI;
        this->RO = RO;
        this->CO = CO;
    }

public:

    // Input tensor
    vector<MatrixD> inp;

    // Output tensor
    vector<MatrixD> out;

    // Virtual Destructor
    virtual ~Layer(){}

    // Trivial getters
    int getInputDepth() {return I;}
    int getOutputDepth() {return O;}
    int getInputRows() {return RI;}
    int getInputCols() {return CI;}
    int getOutputRows() {return RO;}
    int getOutputCols() {return CO;}

    // Forward propagation
    virtual vector<MatrixD> forward(
        vector<MatrixD>& input_tensor) = 0;
    
    // Backward propagation
    virtual vector<MatrixD> backward(
        vector<MatrixD>& output_gradient, T learning_rate) = 0;

};

#endif // LAYER_HPP