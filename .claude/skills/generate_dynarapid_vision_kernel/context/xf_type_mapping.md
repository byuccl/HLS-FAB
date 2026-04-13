# Vitis Vision XF Type to DynaRapid int32 Mapping

DynaRapid only supports `int32_t` for all pointer parameters. This reference table
maps Vitis Vision `XF_*` pixel types to their DynaRapid int32 equivalents and the
**logical pixel range** that must be preserved in `inputs_info` and `behavioral()`.

## Type Conversion Table

| Vitis Type | DynaRapid Type | numpy dtype | Logical Range | Channels | Action |
|---|---|---|---|---|---|
| `XF_8UC1` | `int32_t` | `int32` | 0–255 | 1 | Convert; clamp to [0, 255] |
| `XF_8UC3` | `int32_t` | `int32` | 0–255 | 3 | Convert; clamp each channel to [0, 255] |
| `XF_8UC4` | `int32_t` | `int32` | 0–255 | 4 | Convert; clamp each channel to [0, 255] |
| `XF_16UC1` | `int32_t` | `int32` | 0–65535 | 1 | Convert; clamp to [0, 65535] |
| `XF_16SC1` | `int32_t` | `int32` | -32768–32767 | 1 | Convert; clamp to [-32768, 32767] |
| `XF_16SC3` | `int32_t` | `int32` | -32768–32767 | 3 | Convert; clamp each channel |
| `XF_32SC1` | `int32_t` | `int32` | -2³¹–2³¹-1 | 1 | Native; no clamping needed |
| `XF_32FC1` | **SKIP** | — | — | — | Floating-point not supported |
| `XF_64FC1` | **SKIP** | — | — | — | Floating-point not supported |
| `ap_fixed` | **SKIP** | — | — | — | Fixed-point not supported |
| `ap_ufixed` | **SKIP** | — | — | — | Fixed-point not supported |
| `float` | **SKIP** | — | — | — | Floating-point not supported |
| `double` | **SKIP** | — | — | — | Floating-point not supported |

## Logical Range Constants

Define these in every `generate.py` to document the original pixel semantics:

```python
# For XF_8UC1 / XF_8UC3 origins:
PIXEL_MIN = 0
PIXEL_MAX = 255

# For XF_16UC1 origins:
PIXEL_MIN = 0
PIXEL_MAX = 65535

# For XF_16SC1 origins (e.g. Sobel gradients):
PIXEL_MIN = -32768
PIXEL_MAX = 32767

# For XF_32SC1 origins (native int32):
PIXEL_MIN = -2**31
PIXEL_MAX = 2**31 - 1
```

## inputs_info Value Ranges

The `min` and `max` fields in `inputs_info` must use the **logical pixel range**,
not the full int32 range:

| Original Type | `min` | `max` |
|---|---|---|
| `XF_8UC1` / `XF_8UC3` | 0 | 255 |
| `XF_16UC1` | 0 | 65535 |
| `XF_16SC1` | -32768 | 32767 |
| `XF_32SC1` | -2**31 | 2**31 - 1 |

This ensures `generate_examples()` produces test data in the meaningful range,
not random values across the full int32 space.

## Function Signature

All pointer parameters use `int32_t*` regardless of the original Vitis type:

```c
// Original Vitis:
void boxfilter_accel(ap_uint<8>* img_inp, ap_uint<8>* img_out)

// DynaRapid int32 equivalent:
void boxfilter_accel_int32(int32_t* img_inp, int32_t* img_out)
```

## Shape and total_elements

Shape reflects the logical image dimensions, same as the Vitis version:

```python
# Single-channel image:
"shape": (HEIGHT, WIDTH),
"total_elements": HEIGHT * WIDTH,
"array_dimension": 2,

# 3-channel image (XF_8UC3):
"shape": (HEIGHT, WIDTH, 3),
"total_elements": HEIGHT * WIDTH * 3,
"array_dimension": 3,
```

## behavioral() Output Clamping

Always clamp the output to the logical pixel range before returning:

```python
# For uint8-origin kernels:
return np.clip(result, 0, 255).astype(np.int32).flatten()

# For int16-origin kernels (e.g. Sobel output):
return np.clip(result, -32768, 32767).astype(np.int32).flatten()

# For native int32 kernels:
return result.astype(np.int32).flatten()
```
