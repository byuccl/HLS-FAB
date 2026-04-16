# Dynamatic Architecture Patterns for Vision Kernels

This guide describes how to structure different categories of vision algorithms for optimal Dynamatic HLS synthesis. Dynamatic excels at generating dataflow-optimized hardware, but different algorithm families benefit from different code structures.

The key principle: **structure the code to match the algorithm's data dependencies, not to hide them.**

---

## 1. Stencil-Type Operations (Box Filters, Convolutions, Morphological Ops)

**Category:** Box filter, Gaussian filter, Sobel, Scharr, dilation, erosion, median filter

**Dynamatic strength:** These have regular, predictable neighborhood access patterns that map naturally to dataflow hardware.

**Recommended structure:**
- **Flat loop nests:** Two nested loops (row, then column) with simple loop bodies
- **Uniform memory access:** Each output pixel depends on a fixed-size window of inputs
- **Zero-padding for borders:** Use explicit zero-padding before processing to avoid conditional logic
- **Sequential output:** Write results immediately to output buffer in row-major order

**Code pattern:**
```python
# Pad input with zeros to handle borders
padded = np.pad(img, pad_size, mode='constant', constant_values=0)

# Two simple nested loops — ideal for Dynamatic
out = np.zeros((HEIGHT, WIDTH), dtype=...)
for i in range(HEIGHT):
    for j in range(WIDTH):
        # Extract neighborhood and compute
        window = padded[i:i+FILTER_SIZE, j:j+FILTER_SIZE]
        out[i, j] = window.sum() // (FILTER_SIZE * FILTER_SIZE)
```

**Why this works:** Dynamatic can unroll both loops and pipeline the neighborhood extraction, creating a systolic-like dataflow where each output pixel flows through a fixed computation pattern.

---

## 2. Histogram-Based Algorithms (Histogram Equalization, Back-Projection, calcHist)

**Category:** Histogram equalization, adaptive histogram equalization (CLAHE), histogram back-projection, color reduction

**Dynamatic strength:** Dynamatic's Load-Store Queue (LSQ) can handle the data-dependent indirection (hist[pixel_value]++), but structure matters.

**Recommended structure:**
- **Separate accumulation from lookup:** Split into two distinct phases
  - **Phase 1 (accumulation):** One loop that reads pixels and increments histogram bins
  - **Phase 2 (lookup/apply):** Separate loop that reads pixels and applies the lookup table to produce output
- **LSQ-friendly indexing:** Use direct array indexing (e.g., `hist[pixel_value] += 1`) rather than conditional branches
- **Normalize in between:** Compute derived values (CDF, LUT) between phases so Dynamatic doesn't need to reason across both phases

**Code pattern (histogram equalization):**
```python
# Phase 1: Accumulate histogram
hist = np.zeros(256, dtype=np.int32)
for i in range(HEIGHT):
    for j in range(WIDTH):
        pixel = img[i, j]
        hist[pixel] += 1

# Compute CDF for normalization
cdf = np.cumsum(hist) * 255 // (HEIGHT * WIDTH)

# Phase 2: Apply equalization
out = np.zeros((HEIGHT, WIDTH), dtype=...)
for i in range(HEIGHT):
    for j in range(WIDTH):
        pixel = img[i, j]
        out[i, j] = cdf[pixel]  # LUT lookup
```

**Why this works:** By separating accumulation and lookup, Dynamatic doesn't need to reason about the read-after-write hazard across both phases. Each phase has independent loop iterations (in the output dimension), allowing pipelining and parallelism within each phase.

---

## 3. Reduction Operations (minMaxLoc, sum, mean, stddev)

**Category:** Sum, mean, standard deviation, min/max location, moment computations

**Dynamatic strength:** Dynamatic can parallelize partial reductions across tiles or rows, merging results in a second pass.

**Recommended structure:**
- **Partial reductions:** Compute row-wise or tile-wise accumulators in the first loop
- **Sequential loop counter:** Use a simple counter variable for partial results
- **Final merge:** Combine partial results in a second pass with minimal logic
- **Avoid early termination:** Structure so all iterations complete (Dynamatic prefers fixed iteration counts)

**Code pattern (mean/stddev):**
```python
# Phase 1: Compute per-row partial sums and sum-of-squares
row_sums = np.zeros(HEIGHT, dtype=np.int64)
row_sq_sums = np.zeros(HEIGHT, dtype=np.int64)

for i in range(HEIGHT):
    for j in range(WIDTH):
        pixel = img[i, j]
        row_sums[i] += pixel
        row_sq_sums[i] += pixel * pixel

# Phase 2: Final reduction (merge partial results)
total_sum = np.sum(row_sums)
total_sq_sum = np.sum(row_sq_sums)

mean = total_sum // (HEIGHT * WIDTH)
variance = (total_sq_sum // (HEIGHT * WIDTH)) - (mean * mean)
stddev = np.sqrt(variance)
```

**Why this works:** By computing partial reductions first, Dynamatic can parallelize the outer row loop, then use a simple tree-reduction for the final merge. This gives more opportunities for hardware parallelism than a single sequential accumulator.

---

## 4. Multi-Pass Algorithms (Canny Edge Detection, Morphological Complex Operations)

**Category:** Canny (blur → gradient → non-max suppression → hysteresis), complex morphological sequences

**Dynamatic strength:** Dynamatic can synthesize each pass independently, then decide whether to materialize intermediate results or fuse passes based on available bandwidth and memory.

**Recommended structure:**
- **Explicit intermediate arrays:** Materialize outputs from each pass as full arrays (not streamed)
- **Clear pass boundaries:** Each pass is a separate top-level function or clearly demarcated loop region
- **Independent loop nests:** No data forwarding between passes — each reads from previous pass's output
- **Avoid loop fusion:** Let Dynamatic decide fusion; explicit separation makes the dataflow clearer

**Code pattern (Canny-like edge detection):**
```python
# Pass 1: Gaussian blur
blurred = gaussian_blur(img, SIGMA)

# Pass 2: Compute gradients (Gx, Gy)
gx, gy = sobel_gradients(blurred)
magnitude = np.sqrt(gx*gx + gy*gy)

# Pass 3: Non-maximum suppression
suppressed = non_max_suppress(magnitude, gx, gy)

# Pass 4: Hysteresis thresholding
edges = hysteresis_threshold(suppressed, LOW_THRESH, HIGH_THRESH)
return edges
```

Each pass is written as a complete numpy function. Dynamatic synthesizes each pass independently and can decide at compile time whether to pipeline, fuse, or parallelize.

**Why this works:** Explicit intermediate arrays and clear pass boundaries let Dynamatic see the full dataflow graph. It can then apply optimizations like loop fusion, pipelining, or resource sharing without ambiguity.

---

## 5. Algorithms with Variable Iteration Counts (Feature Matching, RANSAC, Connected Components)

**Category:** Feature matching, connected-component labeling (with flood-fill), RANSAC, any algorithm with data-dependent termination

**Dynamatic strength:** Unlike static HLS tools, **Dynamatic excels here.** It can synthesize variable-length loops and data-dependent control flow efficiently.

**Recommended structure:**
- **Embrace data-dependent control:** Don't artificially constrain to fixed iteration counts
- **Clear loop termination condition:** Use a simple boolean flag or iteration counter with clear exit condition
- **Separate fixed from variable work:** If part of the computation is fixed and part is variable, structure them in separate phases
- **Avoid nested variable loops:** Keep variable-iteration loops as outermost; nest fixed-length loops inside

**Code pattern (connected components):**
```python
# Phase 1: Label connected components (variable iterations due to flood-fill)
labels = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
current_label = 0

for i in range(HEIGHT):
    for j in range(WIDTH):
        if img[i, j] > 0 and labels[i, j] == 0:
            # Flood-fill (variable number of steps)
            current_label += 1
            stack = [(i, j)]
            while len(stack) > 0:  # Variable loop — Dynamatic handles this well
                y, x = stack.pop()
                if 0 <= y < HEIGHT and 0 <= x < WIDTH and labels[y, x] == 0 and img[y, x] > 0:
                    labels[y, x] = current_label
                    # Add neighbors
                    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                        stack.append((y+dy, x+dx))

# Phase 2: (Optional) Relabel consecutively, or output directly
return labels
```

**Why this works:** Dynamatic's synthesis engine can analyze the data dependencies in variable-length loops and generate state machines and queues to handle them. This is actually a strength compared to static synthesis.

---

## 6. Kernels NOT Well-Suited for Dynamatic

Some algorithms don't map well to dataflow hardware regardless of structure:

- **Algorithms with true dynamic memory allocation** (e.g., building arbitrary graphs or trees)
- **Recursive algorithms** without clear base case or bounded depth
- **Algorithms with pointer-chasing** through unknown data structures
- **Complex control flow with many exception paths**

If a kernel falls into these categories, **SKIP IT** with reason "Algorithm structure not suitable for Dynamatic dataflow synthesis."

---

## Writing Descriptions for Dynamatic Kernels

When writing the `description` string in `generate.py`, include a brief note about the **algorithm category** and **why the code structure is chosen**:

### Template:

```python
description = f"""
This HLS kernel performs <algorithm_name>, a <category> operation.

Algorithm category: <Stencil-type / Histogram-based / Reduction / Multi-pass / Variable-iteration>

Structure: <Briefly describe the code structure and why it suits Dynamatic synthesis>
- <Detail 1: e.g., "Two-pass structure: histogram accumulation, then LUT lookup">
- <Detail 2: e.g., "Flat loop nests with zero-padded border handling">
- <Detail 3: e.g., "Partial reductions across rows, merged in a second phase">

Parameters:
    img_inp: Pointer to input image array
    img_out: Pointer to output image array

The kernel processes the image as <description of algorithm without mentioning Dynamatic, HLS, or dataflow>.
Border pixels are handled by <border strategy>.
Output values are <range and how they're computed>.
"""
```

### Example (Box Filter):

```python
description = f"""
This HLS kernel applies a box filter (mean filter) to a grayscale image.

Algorithm category: Stencil-type operation

Structure: Flat loop nest with zero-padded border handling. Each output pixel is computed from a {FILTER_WIDTH}x{FILTER_WIDTH} neighborhood with zero-padding to handle edges uniformly.

Parameters:
    img_inp: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image (row-major).
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
             the filtered output image (same dimensions as input).

Each output pixel is the integer average of its {FILTER_WIDTH}x{FILTER_WIDTH} neighborhood,
including zero-padded border pixels. The average is computed as integer division (floor).
Output values are in the range 0–255, saturated to uint8.
"""
```

---

## Summary: Choosing the Right Pattern

| Algorithm Category | Pattern | Key Characteristic |
|---|---|---|
| Stencil ops | Flat nested loops, zero-padding | Regular neighborhood access |
| Histogram | Two-phase (accumulate, then lookup) | Data-dependent indirection separated |
| Reductions | Partial reductions, then merge | Parallel accumulation + sequential combine |
| Multi-pass | Explicit intermediate arrays | Clear pass boundaries |
| Variable-iteration | Data-dependent loops, clear exit | Lean into irregularity, not away |

Choose the pattern that **honestly reflects the algorithm's data dependencies**. Dynamatic is designed to synthesize whatever structure you describe; hiding or obscuring dependencies doesn't improve synthesis, it makes it harder for Dynamatic to optimize.

