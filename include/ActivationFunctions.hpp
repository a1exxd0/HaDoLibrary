#ifndef ACTIVATION_FUNCTIONS_HPP
#define ACTIVATION_FUNCTIONS_HPP

#include <cmath>


/**
 * @brief ReLU activation function
 * 
 * @tparam T Data type (float, double, long double) (optional)
*/
template<typename T=float, typename = 
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
struct relu {
    T operator()(T x) const {
        return x > 0 ? x : 0;
    }
};

/**
 * @brief Derivative of ReLU activation function
 * 
 * @tparam T Data type (float, double, long double) (optional)
*/
template<typename T=float, typename = 
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
struct relu_prime {
    T operator()(T x) const {
        return x > 0 ? 1 : 0;
    }
};

/**
 * @brief Sigmoid activation function
 * 
 * @tparam T Data type (float, double, long double) (optional)
*/
template<typename T=float, typename = 
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
struct sigmoid {
    T operator()(T x) const {
        return 1 / (1 + exp(-x));
    }
};

/**
 * @brief Derivative of sigmoid activation function
 * 
 * @tparam T Data type (float, double, long double) (optional)
*/
template<typename T=float, typename = 
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
struct sigmoid_prime {
    T operator()(T x) const {
        return x * (1 - x);
    }
};

/**
 * @brief Tanh activation function (namespace collision with cmath)
 * 
 * @tparam T Data type (float, double, long double) (optional)
*/
template<typename T=float, typename = 
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
struct f_tanh {
    T operator()(T x) const {
        return std::tanh(x);
    }
};

/**
 * @brief Derivative of tanh activation function
 * 
 * @tparam T Data type (float, double, long double) (optional)
*/
template<typename T=float, typename = 
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
struct f_tanh_prime {
    T operator()(T x) const {
        T tanhx = std::tanh(x);
        return 1 - tanhx * tanhx;
    }
};

#endif