# Vitis Vision XF Type to NumPy Mapping

This reference table maps Vitis Vision `XF_*` pixel types (used in `xf_config_params.h`)
to their numpy equivalents for use in `inputs_info` and `behavioral()`.

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
