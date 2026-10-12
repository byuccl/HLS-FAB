#ifndef UTILS_HPP
#define UTILS_HPP

#include <ap_int.h>    // For ap_int types
#include <ap_float.h>  // For ap_float types
#include <cmath>       // For std::abs, std::max
#include <limits>      // For std::numeric_limits (for standard types)
#include <iostream>    // For std::cout (for debugging, remove in production code)
#include <type_traits> // For std::is_floating_point, std::is_integral, std::enable_if


// template<typename T>
// struct is_ap_int : std::false_type {};
// template<int W>
// struct is_ap_int<ap_int<W>> : std::true_type {};

template<typename T>
struct is_ap_float : std::false_type {};
template<int W, int E>
struct is_ap_float<ap_float<W, E>> : std::true_type {};


const float AP_FLOAT_BFLOAT16_DEFAULT_EPSILON = 0.03f; //For bfloat16 comparison tolerance

template<typename T>
T local_abs(T value) {
    if constexpr (std::is_floating_point<T>::value) {
        return std::abs(value);
    } else {
        return value < 0 ? -value : value; // For integral types
    }
}

// Helper function for float/double comparison with tolerance
template<typename T>
bool compare_numeric(T actual, T expected, T epsilon = std::numeric_limits<T>::epsilon() * 1000) {
    uint32_t numBits = sizeof(T) * 8;
    // Determine the default epsilon based on type
    T default_epsilon;
    if constexpr (std::is_floating_point<T>::value) {
        default_epsilon = std::numeric_limits<T>::epsilon() * 1000;
    } else if constexpr (is_ap_float<T>::value) {
        default_epsilon = AP_FLOAT_BFLOAT16_DEFAULT_EPSILON;
    } else {
        default_epsilon = T(); // For integral types, epsilon is 0
    }
    // Check if T is a floating-point type
    if constexpr (std::is_floating_point<T>::value) {
        T scale = std::max(static_cast<T>(1.0),
                   std::max(std::abs(actual), std::abs(expected)));
        std::cout << "Comparing floating-point numbers: actual = " << actual 
                  << ", expected = " << expected 
                  << ", epsilon = " << epsilon 
                  << ", scale = " << scale 
                  << ", diff = " << local_abs(actual - expected) << std::endl;
        return local_abs(actual - expected) < default_epsilon * scale;
    }
    // 2. Handle ap_float types
    else if constexpr (is_ap_float<T>::value) {
        if constexpr (T::width == 16 && T::exponent_width == 8) {
            default_epsilon = static_cast<T>(AP_FLOAT_BFLOAT16_DEFAULT_EPSILON);
        }
        T scale = std::max(static_cast<T>(1.0),
                      std::max(local_abs(actual), local_abs(expected)));
        std::cout << "Comparing ap_float numbers: actual = " << actual
                  << ", expected = " << expected
                  << ", default_epsilon = " << default_epsilon
                  << ", scale = " << scale
                  << ", diff = " << local_abs(actual - expected) << std::endl;
        return local_abs(actual - expected) < default_epsilon * scale;
    }
    else {
        // For non-floating-point types (integers, etc.), +-1 tolerance
        // TODO: this was added after the NPUEval results were gathered (which used an exact-match
        // criterion) to accommodate the Vitis Vision dataset's multi-step kernels. It is NOT
        // dataset-aware and currently applies to all integer outputs, including NPUEval. If NPUEval
        // needs to go back to strict exact-match, this must be changed manually (e.g. by passing
        // epsilon=0 explicitly for that dataset, or adding dataset-specific dispatch).
        return (actual >= expected) ? (actual - expected <= T(1)) : (expected - actual <= T(1));
    }
}

#endif // UTILS_HPP