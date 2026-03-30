import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 1080
WIDTH = 1920
XF_MAX_OBJECTS = 10
XF_MAX_ITERS = 4
XF_MAX_OBJ_HEIGHT = 250
XF_MAX_OBJ_WIDTH = 250
TOTAL_BINS = 512

function_signature = (
    "void mean_shift_accel("
    "ap_uint<32>* img_inp, "
    "uint16_t* tlx, "
    "uint16_t* tly, "
    "uint16_t* obj_height, "
    "uint16_t* obj_width, "
    "uint16_t* track, "
    "uint16_t* dx, "
    "uint16_t* dy)"
)

description = f"""
This HLS kernel performs mean-shift object tracking on a {HEIGHT}x{WIDTH} RGBA image.

Parameters:
    img_inp:    Pointer to a flat array of {HEIGHT * WIDTH} uint32 pixel values representing
                a {HEIGHT}x{WIDTH} RGBA input image in row-major order. Each 32-bit value
                packs [R(7:0), G(15:8), B(23:16), A(31:24)].
    tlx:        Pointer to an array of {XF_MAX_OBJECTS} uint16 values representing the
                top-left x-coordinates of each object's bounding box.
    tly:        Pointer to an array of {XF_MAX_OBJECTS} uint16 values representing the
                top-left y-coordinates of each object's bounding box.
    obj_height: Pointer to an array of {XF_MAX_OBJECTS} uint16 values representing the
                height of each object's bounding box.
    obj_width:  Pointer to an array of {XF_MAX_OBJECTS} uint16 values representing the
                width of each object's bounding box.
    dx:         Pointer to an array of {XF_MAX_OBJECTS} uint16 values for the output
                new center x-coordinates of each tracked object.
    dy:         Pointer to an array of {XF_MAX_OBJECTS} uint16 values for the output
                new center y-coordinates of each tracked object.
    track:      Pointer to an array of {XF_MAX_OBJECTS} uint16 values for tracking status
                (1=active, 0=lost/inactive). Objects with track[i]=0 produce dx[i]=dy[i]=0.

The kernel has a scalar parameter frame_status (uint8, NOT in the pointer signature):
  - frame_status=0 (setup/initialization, first frame):
      Computes and stores the model color histogram Qu[i][512] for each active object
      using an Epanechnikov kernel-weighted histogram over the ROI pixels.
      Outputs: dx[i] = (obj_width[i]>>1) + tlx[i]  (initial center x)
               dy[i] = (obj_height[i]>>1) + tly[i]  (initial center y)
      The Qu histograms are retained in static storage for subsequent tracking frames.
      track[] values are unchanged.

  - frame_status=1 (tracking, subsequent frames):
      For each active object, performs {XF_MAX_ITERS} mean-shift iterations:
        1. Compute current histogram Pu over the current ROI window.
        2. For each ROI pixel compute weight = xFTrackmulSqrtLut[Qu_bin / Pu_bin]
           (integer sqrt ratio), masked by Epanechnikov kernel (K <= 256 in Q0.8).
        3. New center = old center + (weighted sum of pixel offsets) / (sum of weights).
        4. Update ROI top-left: x1 = C_x - h_x, y1 = C_y - h_y.
      Outputs: dx[i] = new center x, dy[i] = new center y.
      If the shifted center goes out of image bounds, the center is clamped and
      track[i] is set to 0.

Color histogram (512 bins): bin = (R>>5) + ((G>>5)<<3) + ((B>>5)<<6).

Epanechnikov kernel (Q0.8 fixed-point):
    K = ((x_off^2 + y_off^2) * wh) >> 8   where wh = (1<<16)/(h_x * h_y)
    distance = 256 - K  if K <= 256 else 0

Weight function (uses integer square root LUT xFTrackmulSqrtLut[100]):
    weight = sqrt_lut(Qu[bin]>>8 / Pu[bin]>>8)  if K<=256 and both histograms nonzero
             else 0

The behavioral model is configured for frame_status=0 (FRAME_STATUS=0 constant).
"""

inputs_info = [
    {
        "name": "img_inp",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint32",
        "iotype": "in",
        "min": 0,
        "max": 2**32 - 1,
    },
    {
        "name": "tlx",
        "array_dimension": 1,
        "shape": (XF_MAX_OBJECTS,),
        "total_elements": XF_MAX_OBJECTS,
        "dtype": "uint16",
        "iotype": "in",
        "min": 0,
        "max": WIDTH - 1,
    },
    {
        "name": "tly",
        "array_dimension": 1,
        "shape": (XF_MAX_OBJECTS,),
        "total_elements": XF_MAX_OBJECTS,
        "dtype": "uint16",
        "iotype": "in",
        "min": 0,
        "max": HEIGHT - 1,
    },
    {
        "name": "obj_height",
        "array_dimension": 1,
        "shape": (XF_MAX_OBJECTS,),
        "total_elements": XF_MAX_OBJECTS,
        "dtype": "uint16",
        "iotype": "in",
        "min": 22,
        "max": XF_MAX_OBJ_HEIGHT,
    },
    {
        "name": "obj_width",
        "array_dimension": 1,
        "shape": (XF_MAX_OBJECTS,),
        "total_elements": XF_MAX_OBJECTS,
        "dtype": "uint16",
        "iotype": "in",
        "min": 22,
        "max": XF_MAX_OBJ_WIDTH,
    },
    {
        "name": "track",
        "array_dimension": 1,
        "shape": (XF_MAX_OBJECTS,),
        "total_elements": XF_MAX_OBJECTS,
        "dtype": "uint16",
        "iotype": "in",
        "min": 0,
        "max": 1,
    },
    {
        "name": "dx",
        "array_dimension": 1,
        "shape": (XF_MAX_OBJECTS,),
        "total_elements": XF_MAX_OBJECTS,
        "dtype": "uint16",
        "iotype": "out",
    },
    {
        "name": "dy",
        "array_dimension": 1,
        "shape": (XF_MAX_OBJECTS,),
        "total_elements": XF_MAX_OBJECTS,
        "dtype": "uint16",
        "iotype": "out",
    },
]


# Square LUT for kernel computation (mirrors xFTrackmulKernelLut[200])
_KERNEL_LUT = np.arange(200, dtype=np.int32) ** 2

# Square root LUT for weight computation (mirrors xFTrackmulSqrtLut[100])
_SQRT_LUT = np.array([
    0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9], dtype=np.uint8)


def _xf_sqrt(temp):
    """Matches xFTrackmulSqrt: sqrt from LUT, returns 10 if temp >= 100."""
    if temp < 100:
        return int(_SQRT_LUT[temp])
    return 10


def _xf_inverse_wh(h_x, h_y):
    """Compute wh = (1/(h_x * h_y)) in Q0.16 format, matching Vitis xf::cv::Inverse."""
    # Avoid division by zero
    if h_x == 0 or h_y == 0:
        return 0
    # wh in Q0.16 = (1 << 16) / (h_x * h_y)
    return int((1 << 16) / (h_x * h_y))


def _find_bin(R, G, B):
    """Compute histogram bin index from RGB."""
    r = int(R) >> 5
    g = int(G) >> 5
    b = int(B) >> 5
    return r + (g << 3) + (b << 6)


def _compute_hist_and_bin(img_rgba, x1, y1, obj_hgt, obj_wdt, frame_status,
                          Qu, Pu):
    """Compute histogram for ROI, store in Qu (frame_status=0) or Pu (frame_status=1).
    Also returns the BIN array for the ROI.
    Matches xFTrackmulFindhist / xFTrackmulHist logic.
    Returns: BIN array shape (obj_hgt, obj_wdt)
    """
    h_y = obj_hgt >> 1
    h_x = obj_wdt >> 1
    wh = _xf_inverse_wh(h_x, h_y)  # Q0.16

    hist = np.zeros(TOTAL_BINS, dtype=np.int32)
    BIN = np.zeros((obj_hgt, obj_wdt), dtype=np.uint16)

    for i in range(obj_hgt):
        row_idx = y1 + i
        if row_idx < 0 or row_idx >= HEIGHT:
            continue
        for j in range(obj_wdt):
            col_idx = x1 + j
            if col_idx < 0 or col_idx >= WIDTH:
                continue
            pixel = int(img_rgba[row_idx * WIDTH + col_idx])
            R = pixel & 0xFF
            G = (pixel >> 8) & 0xFF
            B = (pixel >> 16) & 0xFF
            bin_idx = _find_bin(R, G, B)
            BIN[i, j] = bin_idx

            # Epanechnikov kernel distance
            a = i - h_y
            b = j - h_x
            y_abs = abs(a)
            x_abs = abs(b)
            xx = int(_KERNEL_LUT[x_abs]) if x_abs < 200 else x_abs * x_abs
            yy = int(_KERNEL_LUT[y_abs]) if y_abs < 200 else y_abs * y_abs
            K_val = ((xx + yy) * wh) >> 8
            if K_val <= 256:
                distance = 256 - K_val
            else:
                distance = 0
            hist[bin_idx] += distance

    if frame_status == 0:
        for k in range(TOTAL_BINS):
            Qu[k] = hist[k]
    else:
        for k in range(TOTAL_BINS):
            Pu[k] = hist[k]

    return BIN


def _compute_weight_displacement(Qu, Pu, BIN, x1, y1, obj_hgt, obj_wdt, C_x, C_y):
    """Compute weighted displacement and update center.
    Matches xFTrackmulWeight logic.
    Returns: (new_C_x, new_C_y, new_x1, new_y1, track_ok)
    """
    h_x = obj_wdt >> 1
    h_y = obj_hgt >> 1
    wh = _xf_inverse_wh(h_x, h_y)  # Q0.16

    total_x = 0
    total_y = 0
    total_w = 0

    # buf_size in weight: ((y1 + (h_y<<1)) - y1) + 1 = h_y*2 + 1
    # but iterates over obj_hgt rows and buf_size cols (not obj_wdt)
    # From source: buf_size = ((y1 + (h_y<<1)) >> 0) - (y1 >> 0) + 1
    buf_size = (y1 + (h_y << 1)) - y1 + 1  # = h_y*2 + 1 = obj_hgt + 1

    for i in range(obj_hgt):
        for j in range(buf_size):
            if j >= obj_wdt:
                continue
            bin_idx = int(BIN[i, j])
            A = i - h_y
            B = j - h_x
            y_abs = abs(A)
            x_abs = abs(B)
            if x_abs < 200:
                xx = int(_KERNEL_LUT[x_abs])
            else:
                xx = x_abs * x_abs
            if y_abs < 200:
                yy = int(_KERNEL_LUT[y_abs])
            else:
                yy = y_abs * y_abs
            K_val = ((xx + yy) * wh) >> 8

            if K_val > 256 or Qu[bin_idx] == 0 or (Pu[bin_idx] >> 8) == 0:
                weight = 0
            else:
                a_val = Qu[bin_idx] >> 8
                b_val = Pu[bin_idx] >> 8
                weight = _xf_sqrt(a_val // b_val)

            total_y += weight * A
            total_x += weight * B
            total_w += weight

    if total_w != 0:
        dispx = total_x // total_w
        dispy = total_y // total_w
    else:
        dispx = 0
        dispy = 0

    C_x += dispx
    C_y += dispy
    track_ok = True

    # Boundary clamp (matches C++ boundary check)
    if C_y + h_y >= HEIGHT:
        C_y = HEIGHT - h_y - 1
        track_ok = False
    if C_x + h_x >= WIDTH:
        C_x = WIDTH - h_x - 1
        track_ok = False
    if C_y - h_y <= 0:
        C_y = h_y + 1
        track_ok = False
    if C_x - h_x <= 0:
        C_x = h_x + 1
        track_ok = False

    new_x1 = C_x - h_x
    new_y1 = C_y - h_y
    return C_x, C_y, new_x1, new_y1, track_ok


# frame_status hardcoded to 0 for csim test (setup/initialization mode).
# For frame_status=1, call behavioral with the outputs of a prior frame_status=0 call.
FRAME_STATUS = 0


def behavioral(img_inp, tlx, tly, obj_height, obj_width, track):
    """Mean-shift object tracking behavioral model.

    Implements both modes based on FRAME_STATUS constant:

    frame_status=0 (setup/initialization):
        Computes the model color histogram (Qu) for each tracked object ROI.
        Outputs the initial center coordinates:
            dx[i] = (obj_width[i] >> 1) + tlx[i]
            dy[i] = (obj_height[i] >> 1) + tly[i]
        Track status is unchanged.

    frame_status=1 (tracking):
        Computes the current-frame histogram (Pu) for the object ROI.
        For each of XF_MAX_ITERS=4 iterations:
            1. Compute Pu histogram at current window position.
            2. Compute per-pixel weights = sqrt_lut(Qu_bin / Pu_bin) with
               Epanechnikov kernel masking (K <= 256 in Q0.8 format).
            3. Shift center by weighted mean displacement.
        Outputs the new center coordinates in dx[i], dy[i].
        If the object goes out of bounds, track[i] is set to 0.
        Requires that frame_status=0 was previously called (to populate Qu);
        if Qu is zero (first call), displacement is 0 and dx/dy = initial center.

    Color histogram: 512 bins, bin = (R>>5) + ((G>>5)<<3) + ((B>>5)<<6).
    Epanechnikov kernel: K = (x^2 + y^2) / (h_x * h_y) in Q0.8 format,
        distance = max(0, 256 - K), weighted with pixel offsets from ROI center.
    """
    dx_out = np.zeros(XF_MAX_OBJECTS, dtype=np.uint16)
    dy_out = np.zeros(XF_MAX_OBJECTS, dtype=np.uint16)
    track_out = track.copy()

    for i in range(XF_MAX_OBJECTS):
        if track[i] == 0:
            dx_out[i] = 0
            dy_out[i] = 0
            continue

        x1 = int(tlx[i])
        y1 = int(tly[i])
        obj_wdt = int(obj_width[i])
        obj_hgt = int(obj_height[i])

        C_x = (obj_wdt >> 1) + x1
        C_y = (obj_hgt >> 1) + y1

        if FRAME_STATUS == 0:
            # Setup: compute model histogram and return initial center
            Qu = np.zeros(TOTAL_BINS, dtype=np.int32)
            Pu = np.zeros(TOTAL_BINS, dtype=np.int32)
            _compute_hist_and_bin(img_inp, x1, y1, obj_hgt, obj_wdt, 0, Qu, Pu)
            dx_out[i] = np.uint16(C_x)
            dy_out[i] = np.uint16(C_y)

        else:
            # Tracking: XF_MAX_ITERS iterations of hist+weight
            # Qu would normally be populated from frame_status=0 call;
            # here we simulate by computing it from the same image.
            Qu = np.zeros(TOTAL_BINS, dtype=np.int32)
            Pu = np.zeros(TOTAL_BINS, dtype=np.int32)
            # Populate Qu (model histogram) — in real use, this is static from frame 0
            BIN = _compute_hist_and_bin(img_inp, x1, y1, obj_hgt, obj_wdt, 0, Qu, Pu)
            track_ok = True
            for _ in range(XF_MAX_ITERS):
                # Compute current histogram Pu
                BIN = _compute_hist_and_bin(img_inp, x1, y1, obj_hgt, obj_wdt, 1, Qu, Pu)
                # Compute weights and shift center
                C_x, C_y, x1, y1, track_ok = _compute_weight_displacement(
                    Qu, Pu, BIN, x1, y1, obj_hgt, obj_wdt, C_x, C_y)

            dx_out[i] = np.uint16(C_x)
            dy_out[i] = np.uint16(C_y)
            if not track_ok:
                track_out[i] = 0

    return dx_out.flatten(), dy_out.flatten()


examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
