# Dynarapid Skill Documentation Updates

## Overview

This document summarizes the updates made to the `generate_dynarapid_kernel` skill documentation to clarify the conversion from `generate_vision_kernel` examples and to provide Dynamatic-specific architecture guidance.

---

## Key Updates

### 1. **Clarified the Container Type Approach** (SKILL.md Step 3)

**Problem:** The relationship between the function signature, inputs_info dtype, and behavioral model was unclear.

**Solution:** Updated Step 3 to explain the **container type pattern**:
- **Function signature:** Use `int32_t*` or `uint32_t*` (Dynarapid requirement)
- **inputs_info:** Use container type matching the signature (`"int32"` or `"uint32"`)
- **behavioral():** Cast to/from original Vitis pixel type internally

This clarifies that the function signature and inputs_info use uniform container types for Dynarapid's requirements, while the behavioral function still works with the original pixel types (0–255 for uint8, -32768–32767 for int16, etc.) internally.

### 2. **Created conversion_from_vision.md**

A new context file that explains **how to convert a vision kernel example to a dynarapid example**:

- Shows side-by-side comparisons (before/after)
- Explains the **container type approach**:
  - Function signature: `ap_uint<N>*` → `int32_t*` or `uint32_t*`
  - inputs_info dtype: update to match container type (`"int32"` or `"uint32"`)
  - behavioral(): unchanged logic; casts to original pixel type for processing
- Provides examples for three common patterns:
  - Single-channel filters (boxfilter, uint8 pixels in int32 container)
  - Filters with type conversion (sobelfilter, uint8→int16 pixels in int32 container)
  - Multi-input operations (addweighted, 2×uint8→uint8 pixels in int32 container)

### 3. **Created dynamatic_architecture_patterns.md**

A comprehensive guide for structuring different algorithm categories to optimize Dynamatic synthesis:

**Covers 6 algorithm categories:**

1. **Stencil-type operations** (box filter, Sobel, convolution)
   - Use flat loop nests, zero-padding, sequential output
   - Dynamatic maps these to systolic-like dataflow

2. **Histogram-based algorithms** (histogram equalization, CLAHE)
   - Separate accumulation (Phase 1) from lookup (Phase 2)
   - Allows Dynamatic's LSQ to handle data-dependent indirection efficiently

3. **Reduction operations** (sum, mean, min/max, stddev)
   - Compute partial reductions across rows/tiles (Phase 1)
   - Merge results in a final reduction pass (Phase 2)
   - Enables parallel accumulation

4. **Multi-pass algorithms** (Canny edge detection)
   - Materialize intermediate results as full arrays
   - Clear pass boundaries enable independent synthesis
   - Let Dynamatic decide fusion vs. pipelining

5. **Variable-iteration algorithms** (connected components, RANSAC)
   - Lean into data-dependent control flow, not away
   - Dynamatic excels at variable-length loops

6. **Kernels NOT suited for Dynamatic**
   - Floating-point types (skip)
   - Pointer-chasing through unknown structures (skip)
   - Arbitrary dynamic memory allocation (skip)

**Also includes:**
- Detailed code patterns for each category
- Explanation of why each pattern works with Dynamatic
- Template for writing descriptions that note algorithm category and structure

### 4. **Updated SKILL.md Process Sections**

- **Step 3:** Clarified wrapper approach with concrete code examples and a critical note box
- **Step 5b:** Added Dynamatic-specific note about structuring the behavioral model to match data dependencies
- **Step 8:** Added guidance on reconciling descriptions with Dynamatic architecture (note algorithm category and structure)
- **New section:** "Selecting Appropriate Kernels for Dynarapid" listing good candidates, kernels to skip, and those requiring review

### 5. **Updated xf_type_mapping.md**

- Added header note linking to `conversion_from_vision.md`
- Added a "Dynarapid: Function Signature vs. inputs_info" section with concrete examples showing:
  - Function signature uses `int32_t*`
  - inputs_info uses original types
  - behavioral() unwraps via cast
  - Link to conversion guide

---

## How to Use These Updates

### For Skill Users (When Generating Examples):

1. **Read** `context/conversion_from_vision.md` to understand how to convert vision kernel examples
2. **Refer to** `context/dynamatic_architecture_patterns.md` to structure your algorithm appropriately
3. **Check the updated SKILL.md** (especially Steps 3, 5b, 8) for specific guidance on function signatures and descriptions

### For Updating Examples:

The existing examples (boxfilter, addweighted, sobelfilter) don't need code changes because:
- They already use the correct wrapper pattern (int32_t in signature, original types in behavioral)
- The behavioral functions are already optimal for their algorithm categories

However, their **descriptions should be updated** to mention:
- Algorithm category (stencil-type for boxfilter and sobelfilter, element-wise for addweighted)
- Why the code structure suits Dynamatic (flat loops, zero-padding, etc.)

### For Creating New Examples:

1. Start with a `generate_vision_kernel` example as reference
2. Apply conversion from `context/conversion_from_vision.md` (change function signature only)
3. Structure the behavioral model according to the algorithm category in `context/dynamatic_architecture_patterns.md`
4. Write a description that includes the algorithm category and structure rationale

---

## Key Principles

1. **Container types bridge Dynarapid and Vitis Vision:** int32_t or uint32_t in function signature and inputs_info, original pixel types in behavioral(). This allows Dynarapid (which requires uniform pointer types) to work with Vitis Vision kernels (which use various pixel types).

2. **inputs_info dtype must match function signature:** If signature is `int32_t*`, inputs_info dtype is `"int32"`. If signature is `uint32_t*`, inputs_info dtype is `"uint32"`. This consistency is required.

3. **behavioral() casts to original types internally:** The behavioral function's cast (e.g., `.astype(np.uint8)`) converts from the container type to the original pixel type for processing. The algorithm logic remains unchanged.

4. **Algorithm category matters:** Different categories benefit from different structures. Choose the pattern that matches the algorithm (see dynamatic_architecture_patterns.md).

5. **Skip floating-point and unsuitable algorithms:** Don't force-fit incompatible algorithms into Dynamatic.

---

## Files Modified/Created

### Modified:
- `SKILL.md` — Steps 3, 5b, 8; added "Selecting Appropriate Kernels"; updated Examples section

### Created:
- `context/conversion_from_vision.md` — Detailed conversion guide with examples
- `context/dynamatic_architecture_patterns.md` — Algorithm category patterns and synthesis guidance
- `context/xf_type_mapping.md` — Added Dynarapid wrapper explanation

### Summary document (this file):
- `UPDATES_SUMMARY.md` — Overview of changes

---

## Next Steps

1. **Update example descriptions** (boxfilter, addweighted, sobelfilter) to include algorithm category and structure notes
2. **Review existing dynarapid examples** to verify they follow conversion_from_vision.md patterns
3. **Use conversion_from_vision.md when creating new examples** — it serves as a checklist

