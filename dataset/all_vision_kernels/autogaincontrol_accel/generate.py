import numpy as np
from hlsfab import PromptConstructor, generate_examples

NUM_CHANNELS = 3
HIST_SIZE = 4096
CLUSTER = 32
GAIN_ARRAY_SIZE = HIST_SIZE // CLUSTER  # 128
PIXEL_DEPTH = 16

function_signature = "void autogaincontrol_accel(ap_uint<32>* histogram, ap_uint<16>* gain)"

description = f"""Auto Gain Control: computes a per-channel gain value from a multi-channel histogram.

The input is a flattened histogram array of shape [{NUM_CHANNELS}][{HIST_SIZE}] stored in row-major order
(channel 0 first, then channel 1, then channel 2), with {NUM_CHANNELS}*{HIST_SIZE} = {NUM_CHANNELS * HIST_SIZE}
unsigned 32-bit entries total.

The output is a gain array of {NUM_CHANNELS} unsigned 16-bit values, one per channel.

Algorithm per channel:
1. Build a gain lookup table of {GAIN_ARRAY_SIZE} entries: gain_array[i] = uint16((1 << {PIXEL_DEPTH}) - i * {CLUSTER}).
   Note: gain_array[0] wraps to 0 because (1 << {PIXEL_DEPTH}) overflows uint16.
2. Group the {HIST_SIZE} histogram bins into {GAIN_ARRAY_SIZE} clusters of {CLUSTER} consecutive bins each.
3. Sum the bin counts within each cluster independently (each cluster's sum = sum of its {CLUSTER} histogram bins only).
4. Track the cluster index with the strictly greatest sum seen so far (update on strictly greater, so ties keep the earlier cluster).
5. The output gain for that channel is gain_array[winning_cluster_index].

The histogram array is indexed in row-major order: histogram[ch * {HIST_SIZE} + bin].
"""

inputs_info = [
    {
        "name": "histogram",
        "dtype": "uint32",
        "array_dimension": 2,
        "shape": (NUM_CHANNELS, HIST_SIZE),
        "total_elements": NUM_CHANNELS * HIST_SIZE,
        "iotype": "in",
        "min": 0,
        "max": 100000,
    },
    {
        "name": "gain",
        "dtype": "uint16",
        "array_dimension": 1,
        "shape": (NUM_CHANNELS,),
        "total_elements": NUM_CHANNELS,
        "iotype": "out",
    },
]


def behavioral(histogram):
    histogram = histogram.astype(np.uint32)
    hist = histogram.reshape(NUM_CHANNELS, HIST_SIZE)

    # Build gain lookup table
    gain_array = np.zeros(GAIN_ARRAY_SIZE, dtype=np.uint16)
    for i in range(GAIN_ARRAY_SIZE):
        val = (1 << PIXEL_DEPTH) - i * CLUSTER
        gain_array[i] = np.uint16(val & 0xFFFF)

    gain = np.zeros(NUM_CHANNELS, dtype=np.uint16)

    for ch in range(NUM_CHANNELS):
        max_count = np.uint32(0)
        for i in range(GAIN_ARRAY_SIZE):
            cluster_sum = np.uint32(0)
            for j in range(CLUSTER):
                cluster_sum += hist[ch, i * CLUSTER + j]
            if cluster_sum > max_count:
                max_count = cluster_sum
                gain[ch] = gain_array[i]

    return gain.astype(np.uint16).flatten()


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
