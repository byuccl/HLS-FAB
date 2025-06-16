#ifndef JSON_APINT_ADAPTER_HPP
#define JSON_APINT_ADAPTER_HPP

#include <ap_int.h>
#include "json.hpp"
#include <cstdint>

using json = nlohmann::json;

// Specialization for ap_int<N>
template<int W>
struct nlohmann::adl_serializer<ap_int<W>> {
    static void to_json(json& j, const ap_int<W>& value) {
        j = static_cast<int64_t>(value); // Use signed int for JSON
    }

    static void from_json(const json& j, ap_int<W>& value) {
        value = static_cast<ap_int<W>>(j.get<int64_t>());
    }
};

// Specialization for ap_uint<N>
template<int W>
struct nlohmann::adl_serializer<ap_uint<W>> {
    static void to_json(json& j, const ap_uint<W>& value) {
        j = static_cast<uint64_t>(value); // Use unsigned int for JSON
    }

    static void from_json(const json& j, ap_uint<W>& value) {
        value = static_cast<ap_uint<W>>(j.get<uint64_t>());
    }
};

#endif // JSON_APINT_ADAPTER_HPP