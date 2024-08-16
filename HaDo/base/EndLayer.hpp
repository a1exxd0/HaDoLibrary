#ifndef END_LAYER_HPP
#define END_LAYER_HPP

#include <Eigen/Dense>
#include <vector>
#include <memory>

using Eigen::Matrix;
using std::vector;
using Eigen::Dynamic;

namespace hado {

/**
 * @brief End layer class for getting error gradient w.r.t results
 * 
 * @tparam T Data type (float for speed, double accuracy) (optional)
*/
template<typename T=float>
class EndLayer {
private:

    // Assert that T is either float, double, or long double at compiler time
    static_assert(
        std::is_same_v<T, float>
        || std::is_same_v<T, double>
        || std::is_same_v<T, long double>,
        "T must be either float, double, or long double."
    );

protected:

    // Convenience typedef
    typedef Matrix<T, Dynamic, Dynamic> MatrixD;

    // Depth of input matrix
    int D;

    // Rows in input matrix
    int R;

    // Columns in input matrix
    int C;

    /**
     * @brief Constructor for basic end layer of neural network
     * 
     * @param D Depth of end node
     * @param R Number of rows of end node
     * @param C Number of columns of end node
    */
    EndLayer(const int D, const int R, const int C) : D(D), R(R), C(C) {}

public:

    // Getters
    [[nodiscard]] int getDepth() const { return D; }
    [[nodiscard]] int getRows() const { return R; }
    [[nodiscard]] int getCols() const { return C; }

    // Copy constructor
    EndLayer(const EndLayer& el) {
        D = el.getDepth();
        R = el.getRows();
        C = el.getCols();
    }

    // Virtual clone
    virtual std::unique_ptr<EndLayer<T>> clone() const = 0;

    // Virtual destructor
    virtual ~EndLayer() = default;

    // Assert input tensor dimensions
    void constexpr assertInputDimensions(const vector<MatrixD> &input_tensor) const {
        if (input_tensor.size() != static_cast<size_t>(D)
            || input_tensor[0].rows() != this->R
            || input_tensor[0].cols() != this->C){
            std::cerr << "Expected depth " << this->D << ", got depth " << input_tensor.size() << endl;
            std::cerr << "Expected rows " << this->R << ", got rows " << input_tensor[0].rows() << endl;
            std::cerr << "Expected cols " << this->C << ", got cols " << input_tensor[0].cols() << endl;
            throw std::invalid_argument("Input tensor match dimensions of layer.");
            }
    }

    // Calculate error w.r.t results
    virtual T forward(
        vector<MatrixD>& res, vector<MatrixD>& true_res) = 0;

    // Calculate error gradient w.r.t results
    virtual vector<MatrixD> backward(
        vector<MatrixD>& res, vector<MatrixD>& true_res) = 0;
};

}

#endif // END_LAYER_HPP
