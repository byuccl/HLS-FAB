import numpy as np
from hlsfab import PromptConstructor, generate_examples

# Number of keypoints (fixed for this configuration)
NUM_KEYPOINTS_1 = 100
NUM_KEYPOINTS_2 = 100
RATIO_THRESH = 0.8
DESC_BITS = 256  # ORB descriptor size in bits

function_signature = "void bfmatcher_accel(int32_t* desc_list1, int32_t* desc_list2, ap_int<16>* desc_match_idx)"

description = f"""Performs brute-force matching of ORB binary descriptors using Hamming distance with Lowe's ratio test.

The function takes two sets of {DESC_BITS}-bit binary descriptors and finds the best match for each
query descriptor in the training set.

Inputs:
- desc_list1: Array of {NUM_KEYPOINTS_1} query descriptors, each a {DESC_BITS}-bit value stored as a
  single ap_uint<{DESC_BITS}> element. Each element encodes a binary ORB descriptor.
- desc_list2: Array of {NUM_KEYPOINTS_2} training descriptors, same format as desc_list1.

Output:
- desc_match_idx: Array of {NUM_KEYPOINTS_1} int16 match indices. For each query descriptor i,
  desc_match_idx[i] contains the index j of the best-matching training descriptor, or -1 if the
  match does not pass the ratio test.

The matching algorithm:
1. For each query descriptor, compute the Hamming distance (number of differing bits) to every
   training descriptor.
2. Track the best match (minimum distance) and second-best match.
3. Apply Lowe's ratio test: if best_distance < {RATIO_THRESH} * second_best_distance, the match
   is accepted and the index of the best match is stored. Otherwise, -1 is stored.

The Hamming distance between two {DESC_BITS}-bit descriptors counts the number of bit positions
where the two descriptors differ (popcount of XOR)."""

inputs_info = [
    {
        "name": "desc_list1",
        "array_dimension": 1,
        "shape": (NUM_KEYPOINTS_1 * (DESC_BITS // 32),),
        "total_elements": NUM_KEYPOINTS_1 * (DESC_BITS // 32),
        "dtype": "int32",
        "iotype": "in",
        "min": 0,
        "max": int(np.iinfo(np.uint32).max),
    },
    {
        "name": "desc_list2",
        "array_dimension": 1,
        "shape": (NUM_KEYPOINTS_2 * (DESC_BITS // 32),),
        "total_elements": NUM_KEYPOINTS_2 * (DESC_BITS // 32),
        "dtype": "int32",
        "iotype": "in",
        "min": 0,
        "max": int(np.iinfo(np.uint32).max),
    },
    {
        "name": "desc_match_idx",
        "array_dimension": 1,
        "shape": (NUM_KEYPOINTS_1,),
        "total_elements": NUM_KEYPOINTS_1,
        "dtype": "int32",
        "iotype": "out",
    },
]


def _hamming_distance_256(desc1_words, desc2_words):
    """Compute Hamming distance between two 256-bit descriptors.
    Each descriptor is represented as 8 uint32 words."""
    dist = 0
    for k in range(8):
        xor_val = int(desc1_words[k]) ^ int(desc2_words[k])
        dist += bin(xor_val & 0xFFFFFFFF).count("1")
    return dist


def behavioral(desc_list1, desc_list2):
    # Reshape descriptors: each descriptor is 256 bits = 8 x uint32
    words_per_desc = DESC_BITS // 32  # 8
    descs1 = desc_list1.reshape(NUM_KEYPOINTS_1, words_per_desc)
    descs2 = desc_list2.reshape(NUM_KEYPOINTS_2, words_per_desc)

    match_idx = np.full(NUM_KEYPOINTS_1, -1, dtype=np.int16)

    for i in range(NUM_KEYPOINTS_1):
        dist_min_1 = 511  # ap_uint<9> max initial
        dist_min_2 = 511
        best_match = 0

        for j in range(NUM_KEYPOINTS_2):
            local_dist = _hamming_distance_256(descs1[i], descs2[j])

            if local_dist < dist_min_1:
                dist_min_2 = dist_min_1
                dist_min_1 = local_dist
                best_match = j
            elif local_dist < dist_min_2:
                dist_min_2 = local_dist

        # Lowe's ratio test
        if float(dist_min_1) < RATIO_THRESH * float(dist_min_2):
            match_idx[i] = np.int16(best_match)
        else:
            match_idx[i] = np.int16(-1)

    return match_idx.flatten()


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
