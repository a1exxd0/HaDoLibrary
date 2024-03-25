#ifndef ACTIVATION_FUNCTIONS_HPP
#define ACTIVATION_FUNCTIONS_HPP

#include <cmath>

template<typename T, typename = 
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
struct relu {
    T operator()(T x) const {
        return x > 0 ? x : 0;
    }
};

template<typename T, typename = 
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
struct relu_prime {
    T operator()(T x) const {
        return x > 0 ? 1 : 0;
    }
};

template<typename T, typename = 
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
struct sigmoid {
    T operator()(T x) const {
        return 1 / (1 + exp(-x));
    }
};

template<typename T, typename = 
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
struct sigmoid_prime {
    T operator()(T x) const {
        return x * (1 - x);
    }
};

template<typename T, typename = 
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
struct struct_tanh {
    T operator()(T x) const {
        return std::tanh(x);
    }
};

template<typename T, typename = 
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
struct tanh_prime {
    T operator()(T x) const {
        T tanhx = std::tanh(x);
        return 1 - tanhx * tanhx;
    }
};

#endif