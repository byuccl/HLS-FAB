# Vitis Vision XF Type to NumPy Mapping (for Dynarapid)

This reference table maps Vitis Vision `XF_*` pixel types (used in `xf_config_params.h`)
to their container types for use in `inputs_info` and function signatures in Dynarapid kernels.

**Important for Dynarapid:** The **function signature and inputs_info** use container types (`int32_t*` or `uint32_t*`), while the **behavioral function** casts to/from the original Vitis pixel types internally. See `conversion_from_vision.md` for the container approach.

## Pixel Type Mapping

| Vitis Type   | numpy dtype | Channels | Bits/pixel | Notes                              |
|--------------|-------------|----------|------------|-------------------------------------|
| `XF_8UC1`    | `uint8`     | 1        | 8          | Grayscale, most common              |
| `XF_8UC3`    | `uint8`     | 3        | 24         | RGB/BGR 3-channel                   |
| `XF_8UC4`    | `uint8`     | 4        | 32         | RGBA 4-channel                      |
| `XF_16UC1`   | `uint16`    | 1        | 16         | 16-bit unsigned grayscale           |
| `XF_16SC1`   | `int16`     | 1        | 16         | 16-bit signed (common for gradients)|
| `XF_16SC3`   | `int16`     | 3        | 48         | 16-bit signed 3-channel             |
| `XF_32FC1`   | `float32`   | 1        | 32         | 32-bit float                        |
| `XF_32SC1`   | `int32`     | 1        | 32         | 32-bit signed integer               |

## Value Ranges for inputs_info

| numpy dtype | min     | max    |
|-------------|---------|--------|
| `uint8`     | 0       | 255    |
| `uint16`    | 0       | 65535  |
| `int16`     | -32768  | 32767  |
| `int32`     | -2^31   | 2^31-1 |
| `float32`   | -3.0    | 3.0 (application-dependent) |

## INPUT_PTR_WIDTH and ap_uint width

The `INPUT_PTR_WIDTH` define controls the AXI bus width for the image pointer.
This affects the `ap_uint<N>` type in the function signature:

| INPUT_PTR_WIDTH | Function signature type     |
|-----------------|-----------------------------|
| 8               | `ap_uint<8>*`               |
| 16              | `ap_uint<16>*`              |
| 32              | `ap_uint<32>*`              |
| 64              | `ap_uint<64>*`              |
| 128             | `ap_uint<128>*`             |
| 256             | `ap_uint<256>*`             |
| 512             | `ap_uint<512>*`             |

Always use the actual `INPUT_PTR_WIDTH` value from `xf_config_params.h` in the function signature.

## Shape and total_elements

For single-channel images:
```python
"shape": (HEIGHT, WIDTH),
"total_elements": HEIGHT * WIDTH,
"array_dimension": 2,
```

For 3-channel images (XF_8UC3):
```python
"shape": (HEIGHT, WIDTH, 3),
"total_elements": HEIGHT * WIDTH * 3,
"array_dimension": 3,
```

For combined multi-output (e.g., Sobel Gx + Gy):
```python
"shape": (2, HEIGHT, WIDTH),
"total_elements": 2 * HEIGHT * WIDTH,
"array_dimension": 3,
```

---

## Dynarapid: Container Types in Signature and inputs_info

**Function signature (int32_t or uint32_t container):**
```python
function_signature = "void boxfilter_accel(int32_t* img_inp, int32_t* img_out)"
# OR
function_signature = "void boxfilter_accel(uint32_t* img_inp, uint32_t* img_out)"
```

**inputs_info (matches container type):**
```python
inputs_info = [
    {
        "name": "img_inp",
        "dtype": "int32",      # Matches int32_t* signature (or "uint32" for uint32_t*)
        ...
    },
    {
        "name": "img_out",
        "dtype": "int32",      # Matches int32_t* signature (or "uint32" for uint32_t*)
        ...
    },
]
```

**behavioral() (casts to/from original pixel type):**
```python
def behavioral(img_inp):
    # Cast from container type (int32/uint32) to original Vitis pixel type
    img = img_inp.reshape((HEIGHT, WIDTH)).astype(np.uint8)  # Original type from XF_8UC1
    # ... process using original pixel type (uint8) ...
    return out.astype(np.uint8).flatten()  # Cast back via container if needed
```

The function signature and inputs_info use container types (`int32_t*` or `uint32_t*`) because Dynarapid requires uniform pointer types. The behavioral function casts to the original Vitis pixel type (uint8, int16, etc.) for processing, which is the internal logic. This lets Dynarapid work with Vitis Vision kernels while maintaining the correct pixel value ranges and semantics.

For more details, see `conversion_from_vision.md`.
