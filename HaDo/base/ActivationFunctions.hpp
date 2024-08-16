#ifndef ACTIVATION_FUNCTIONS_HPP
#define ACTIVATION_FUNCTIONS_HPP

#include <cmath>

namespace hado {
    
/**
 * @brief ReLU activation function
 * 
 * @tparam T Data type (float, double, long double) (optional)
*/
template<typename T=float, typename = 
    std::enable_if_t<std::is_arithmetic_v<T>, T>>
struct relu {
    [[nodiscard]] inline T operator()(T x) const {
        return x > 0 ? x : 0;
    }
};

/**
 * @brief Derivative of ReLU activation function
 * 
 * @tparam T Data type (float, double, long double) (optional)
*/
template<typename T=float, typename = 
    std::enable_if_t<std::is_arithmetic_v<T>, T>>
struct relu_prime {
    [[nodiscard]] inline T operator()(T x) const {
        return x > 0 ? 1 : 0;
    }
};

/**
 * @brief Sigmoid activation function
 * 
 * @tparam T Data type (float, double, long double) (optional)
*/
template<typename T=float, typename = 
    std::enable_if_t<std::is_arithmetic_v<T>, T>>
struct sigmoid {
    [[nodiscard]] inline T operator()(T x) const {
        return 1 / (1 + exp(-x));
    }
};

/**
 * @brief Derivative of sigmoid activation function
 * 
 * @tparam T Data type (float, double, long double) (optional)
*/
template<typename T=float, typename = 
    std::enable_if_t<std::is_arithmetic_v<T>, T>>
struct sigmoid_prime {
    [[nodiscard]] inline T operator()(T inp) const {
        sigmoid<T> f;
        T x = f(inp);
        return x * (1 - x);
    }
};

/**
 * @brief Tanh activation function (namespace collision with cmath)
 * 
 * @tparam T Data type (float, double, long double) (optional)
*/
template<typename T=float, typename = 
    std::enable_if_t<std::is_arithmetic_v<T>, T>>
struct f_tanh {
    [[nodiscard]] inline T operator()(T x) const {
        return std::tanh(x);
    }
};

/**
 * @brief Derivative of tanh activation function
 * 
 * @tparam T Data type (float, double, long double) (optional)
*/
template<typename T=float, typename = 
    std::enable_if_t<std::is_arithmetic_v<T>, T>>
struct f_tanh_prime {
    [[nodiscard]] inline T operator()(T x) const {
        T tanhx = std::tanh(x);
        return 1 - tanhx * tanhx;
    }
};

}

#endif // ACTIVATION_FUNCTIONS_HPP
