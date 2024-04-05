#ifndef FLATTENING_LAYER_HPP
#define FLATTENING_LAYER_HPP
#include "Layer.hpp"

template <typename T = float>
class FlatteningLayer : public Layer<T>
{
private:
    typedef Matrix<T, Dynamic, Dynamic> MatrixD; // Private typedef for convenience
public:
    FlatteningLayer(int I, int RI, int CI)
        : Layer<T>(I, 1, RI, CI, 1, I * RI * CI) // Output depth is 1, output rows is 1, output columns is product of input dimensions
    {
    }

    virtual std::unique_ptr<Layer<T>> clone() const override
    {
        return std::make_unique<FlatteningLayer>(*this);
    }

#pragma GCC push_options
#pragma GCC optimize("O2")
    virtual vector<MatrixD> forward(
        vector<MatrixD> &input_tensor) override
    {
        this->assertInputDimensions(input_tensor);

        // Flatten the input tensor
        MatrixD flattened(this->CI * this->RI * this->I, 1);
        size_t currentRow = 0;
        for (size_t i = 0; i < input_tensor.size(); ++i)
        {
            Eigen::Map<MatrixD>(flattened.data() + currentRow, this->RI * this->CI, 1) = Eigen::Map<const MatrixD>(input_tensor[i].data(), this->RI * this->CI, 1);
            currentRow += this->RI * this->CI;
        }

        this->out.clear();
        this->out.push_back(flattened);
        return this->out;
    }
#pragma GCC pop_options

#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC push_options
#pragma GCC optimize("O2")
    virtual vector<MatrixD> backward(
        vector<MatrixD> &output_gradient, const T learning_rate) override
    {
        // For a flattening layer, the backward propagation simply involves reshaping
        // the gradient to match the input tensor's shape. The learning rate is not used

        vector<MatrixD> input_gradient(this->I);
        for (size_t i = 0; i < this->I; ++i)
        {
            input_gradient[i] = Eigen::Map<MatrixD>(output_gradient[0].data() + i * this->RI * this->CI, this->RI, this->CI);
        }

        return input_gradient;
    }
};
#pragma GCC pop_options

#endif // FLATTENING_LAYER_HPP