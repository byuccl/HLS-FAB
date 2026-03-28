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
    track:      Pointer to an array of {XF_MAX_OBJECTS} uint16 values for the tracking
                status of each object (1 = tracking, 0 = lost).

The kernel operates in two modes based on frame_status (scalar param, 0 for first frame):
  - frame_status=0 (setup): Computes the model color histogram for each object ROI using
    an Epanechnikov kernel. Outputs dx[i] = tlx[i] + obj_width[i]//2, dy[i] = tly[i] +
    obj_height[i]//2 (initial centers). track values are unchanged.
  - frame_status=1 (tracking): Iteratively computes current histogram, calculates weights
    from sqrt(model_hist / current_hist), and shifts the object center. Repeats for up to
    {XF_MAX_ITERS} iterations.

Color histogram: Each pixel's RGB channels are quantized to 3 bits each (>>5), yielding
a bin index = (R>>5) + ((G>>5)<<3) + ((B>>5)<<6), for {TOTAL_BINS} bins total. Bins are
weighted by an Epanechnikov kernel: weight = max(0, 1 - (dx^2+dy^2)/(hx*hy)) where
(dx,dy) is the pixel offset from the ROI center and (hx,hy) are the half-dimensions.

For this behavioral model, only frame_status=0 (setup mode) is implemented. Each
trackable object outputs its initial center coordinates.
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


def behavioral(img_inp, tlx, tly, obj_height, obj_width, track):
    """Mean-shift tracking setup (frame_status=0).

    On the first frame, the kernel computes the model histogram for each object
    and returns the initial center coordinates. No displacement is applied.
    """
    dx_out = np.zeros(XF_MAX_OBJECTS, dtype=np.uint16)
    dy_out = np.zeros(XF_MAX_OBJECTS, dtype=np.uint16)

    for i in range(XF_MAX_OBJECTS):
        if track[i] == 0:
            dx_out[i] = 0
            dy_out[i] = 0
            continue

        # On frame_status=0 (setup), the kernel computes the initial center
        # C_x = (obj_wdt >> 1) + x1
        # C_y = (obj_hgt >> 1) + y1
        w = int(obj_width[i])
        h = int(obj_height[i])
        x1 = int(tlx[i])
        y1 = int(tly[i])

        c_x = (w >> 1) + x1
        c_y = (h >> 1) + y1

        dx_out[i] = np.uint16(c_x)
        dy_out[i] = np.uint16(c_y)

    return dx_out.flatten(), dy_out.flatten()


examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
