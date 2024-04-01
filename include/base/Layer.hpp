#ifndef LAYER_HPP
#define LAYER_HPP

#define _MAX_DEPTH_UNTIL_THREADING 1
#define _MAX_PROD_UNTIL_THREADING 2000

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <iostream>
#include <json/json.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;
using nlohmann::json;
using std::cout;
using std::vector;

#ifndef endl
#define endl "\n"
#endif

/**
 * @brief Base layer class. T will only work for float, double,
 * or long double. Can't construct this directly, must derive a
 * child class to use.
 *
 * @tparam T Data type (float for speed, double accuracy) (optional)
 */
template <typename T = float>
class Layer
{
private:
    // Layer dimensions
    int I, O, RI, CI, RO, CO;
    // Assert that T is either float, double, or long double at compiler time
    #pragma GCC diagnostic ignored "-Wparentheses"
    static_assert(
        std::is_same<T, float>::value || std::is_same<T, double>::value || std::is_same<T, long double>::value, "T must be either float, double, or long double.");

    typedef Matrix<T, Dynamic, Dynamic> MatrixD; // Convenience typedef

protected:
    /**
     * @brief Layer constructor to instantiate input and output vectors
     *
     * @param I Depth of input tensor
     * @param O Depth of output tensor
     * @param RI Rows in input tensor
     * @param CI Columns in input tensor
     * @param RO Rows in output tensor
     * @param CO Columns in output tensor
     */
    Layer(int I, int O, int RI, int CI, int RO, int CO) : inp(I), out(O)
    {
        // Assert positivity of dimensions
        assert(I > 0 && O > 0 && RI > 0 && CI > 0 && RO > 0 && CO > 0);

        this->I = I;
        this->O = O;
        this->RI = RI;
        this->CI = CI;
        this->RO = RO;
        this->CO = CO;
    }

    // Copy constructor
    Layer(const Layer &other) : inp(other.inp), out(other.out)
    {
        I = other.getInputDepth();
        O = other.getOutputDepth();
        RI = other.getInputRows();
        CI = other.getInputCols();
        RO = other.getOutputRows();
        CO = other.getOutputCols();
    }

public:
    // Input tensor
    vector<MatrixD> inp;

    // Output tensor
    vector<MatrixD> out;

    // Virtual clone
    virtual std::unique_ptr<Layer<T>> clone() const = 0;

    // Virtual Destructor
    virtual ~Layer() {}

    // Assert input tensor dimensions
    void constexpr assertInputDimensions(const vector<MatrixD> &input_tensor) const {
        if (input_tensor.size() != (size_t) this->I 
            || input_tensor[0].rows() != this->RI
            || input_tensor[0].cols() != this->CI){
            std::cerr << "Expected depth " << this->I << " but got depth " << input_tensor.size() << endl;
            std::cerr << "Expected rows " << this->RI << " but got rows " << input_tensor[0].rows() << endl;
            std::cerr << "Expected cols " << this->CI << " but got cols " << input_tensor[0].cols() << endl;
            throw std::invalid_argument("Input tensor match dimensions of layer.");
        }
    }

    // Assert output tensor dimensions
    void constexpr assertOutputDimensions(const vector<MatrixD> &output_tensor) const {
        if (output_tensor.size() != (size_t) this->O
            || output_tensor[0].rows() != this->RO
            || output_tensor[0].cols() != this->CO){
            std::cerr << "Expected depth " << this->O << " but got depth " << output_tensor.size() << endl;
            std::cerr << "Expected rows " << this->RO << " but got rows " << output_tensor[0].rows() << endl;
            std::cerr << "Expected cols " << this->CO << " but got cols " << output_tensor[0].cols() << endl;
            throw std::invalid_argument("Output tensor match dimensions of layer.");
        }
    }

    // Trivial getters
    int getInputDepth() const { return I; }
    int getOutputDepth() const { return O; }
    int getInputRows() const { return RI; }
    int getInputCols() const { return CI; }
    int getOutputRows() const { return RO; }
    int getOutputCols() const { return CO; }

    // Forward propagation
    virtual vector<MatrixD> forward(
        vector<MatrixD> &input_tensor) = 0;

    // Backward propagation
    virtual vector<MatrixD> backward(
        vector<MatrixD> &output_gradient, const T learning_rate) = 0;
};

#endif // LAYER_HPP