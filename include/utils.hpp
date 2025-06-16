#ifndef UTILS_HPP
#define UTILS_HPP

#include <type_traits>
#include <cmath>
#include <limits>
// Required for std::cerr, std::cout (though not directly used in compare_numeric,
// often useful in a utils header for debugging or error reporting)
#include <iostream> 


// Helper function for float/double comparison with tolerance
template<typename T>
bool compare_numeric(T actual, T expected, T epsilon = std::numeric_limits<T>::epsilon() * 100) {
    // Check if T is a floating-point type
    if (std::is_floating_point<T>::value) {
        std::cout << "Comparing floating-point numbers: " << actual << " vs " << expected << std::endl;
        std::cout << "Difference is " << std::abs(actual - expected) << std::endl;
        T scale = std::max({static_cast<T>(1.0), std::fabs(actual), std::fabs(expected)});
                std::cout << "Using epsilon: " << epsilon*scale << std::endl;

        return std::abs(actual - expected) < epsilon * scale;
    } else {
        // For non-floating-point types (integers, etc.), exact comparison
        return actual == expected;
    }
}

#endif // UTILS_HPP